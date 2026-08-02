/**
 * C++ Levenshtein alignment — insert + delete only (no substitution).
 *
 * Uses LibTorch tensors directly (no Python list serialization overhead).
 *
 * DP recurrence: substitution cost = 2 (i.e. delete + insert).
 * This is equivalent to disallowing substitution and allowing only
 * insertion (cost 1) + deletion (cost 1) + keep-if-equal (cost 0).
 *
 * Three entry points share one DP/backtrack core (align_one_raw):
 *   - levenshtein_align        single-pair raw alignment (deletions + per-gap
 *                              insertion lists) — unchanged behavior.
 *   - levenshtein_deletion_batch   packed boundary-protected deletion masks.
 *   - levenshtein_insertion_batch  packed placeholder oracle + PLH roll-in.
 * The two batch entry points process the whole batch inside one C++ call
 * (amortising the pybind boundary) and return flat packed tensors + [B+1]
 * offsets so only O(B) Python objects cross the boundary.
 */

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace {

// ── Flat-array DP tables (1-D row-major, better cache locality) ──────────
//
// Raw `new[]` instead of `std::vector::resize` so the O(n·m) table is *not*
// zero-initialized first: the DP fill overwrites every (n+1)×(m+1) entry, so
// value-initialization was pure wasted memory traffic (perf: ~4-5% of the
// op's self time went to libc allocation/zeroing).

struct DPResult {
    std::unique_ptr<int[]> d;        // (n+1) × (m+1) flattened, row-major
    std::unique_ptr<uint8_t[]> back; // (n+1) × (m+1) flattened, row-major
    int n, m, stride;

    int  d_at(int i, int j) const { return d[i * stride + j]; }
    int& d_at(int i, int j)       { return d[i * stride + j]; }
    uint8_t  back_at(int i, int j) const { return back[i * stride + j]; }
    uint8_t& back_at(int i, int j)       { return back[i * stride + j]; }
};

DPResult edit_distance2_with_dp(const int64_t* x, int n,
                                const int64_t* y, int m) {
    DPResult result;
    result.n = n;
    result.m = m;
    result.stride = m + 1;
    const size_t size =
        static_cast<size_t>(n + 1) * static_cast<size_t>(m + 1);
    result.d.reset(new int[size]);        // default-init: no zeroing
    result.back.reset(new uint8_t[size]); // default-init: no zeroing

    // Initialize boundaries
    for (int i = 0; i <= n; ++i) {
        result.d_at(i, 0) = i;
        result.back_at(i, 0) = 1;  // delete
    }
    for (int j = 0; j <= m; ++j) {
        result.d_at(0, j) = j;
        result.back_at(0, j) = 2;  // insert
    }

    // DP fill
    for (int i = 1; i <= n; ++i) {
        const int64_t xi = x[i - 1];
        const int row = i * result.stride;
        const int prev_row = (i - 1) * result.stride;
        for (int j = 1; j <= m; ++j) {
            int best_val = std::numeric_limits<int>::max();
            uint8_t best_op = 0;

            // Delete: cost 1
            int cand = result.d[prev_row + j] + 1;
            if (cand < best_val) {
                best_val = cand;
                best_op = 1;
            }

            // Insert: cost 1
            cand = result.d[row + (j - 1)] + 1;
            if (cand < best_val) {
                best_val = cand;
                best_op = 2;
            }

            // Match: cost 0 if equal, can't substitute (cost would be 2)
            if (xi == y[j - 1]) {
                cand = result.d[prev_row + (j - 1)];
                if (cand < best_val) {
                    best_val = cand;
                    best_op = 0;
                }
            }

            result.d[row + j] = best_val;
            result.back[row + j] = best_op;
        }
    }

    return result;
}

// ── Raw alignment result (vectors; tensor conversion happens per-entry) ──

struct RawAlign {
    std::vector<int64_t> deletions;            // indices in y (may include boundaries)
    std::vector<int> surviving;                // non-deleted indices in y
    std::vector<std::vector<int64_t>> per_gap; // tokens per gap, gap g anchored at surviving[g]
};

// Align one pair (raw pointers to int64 buffers). Computes the DP table,
// backtracks once into (deletions, insertions_raw, insertion_after), then
// builds surviving + per-gap insertion lists using the same anchoring rules
// as the pure-Python fallback: insertions before the first surviving token
// are dropped, trailing insertions go to the last gap.
RawAlign align_one_raw(const int64_t* x, int n, const int64_t* y, int m) {
    RawAlign result;

    // ── DP ─────────────────────────────────────────────────────────────
    DPResult dp = edit_distance2_with_dp(x, n, y, m);

    // ── Single backtrack pass: deletions + insertions together ─────────
    std::vector<int64_t> insertions_raw;
    std::vector<int> insertion_after;
    result.deletions.reserve(static_cast<size_t>(n));
    insertions_raw.reserve(static_cast<size_t>(m));
    insertion_after.reserve(static_cast<size_t>(m));

    {
        int i = n, j = m;
        int current_after = n;
        while (i > 0 || j > 0) {
            const uint8_t op = dp.back_at(i, j);
            if (op == 0) {        // match
                --i; --j;
                current_after = i;
            } else if (op == 1) { // delete
                --i;
                result.deletions.push_back(static_cast<int64_t>(i));
            } else {              // insert
                --j;
                insertions_raw.push_back(y[j]);
                insertion_after.push_back(current_after);
            }
        }
        std::reverse(result.deletions.begin(), result.deletions.end());
        std::reverse(insertions_raw.begin(), insertions_raw.end());
        std::reverse(insertion_after.begin(), insertion_after.end());
    }

    // ── Build surviving-set and per-gap insertion lists ────────────────
    // Boolean mask for O(1) deletion lookup (replaces std::set<int>).
    std::vector<uint8_t> del_mask(static_cast<size_t>(n), 0);
    for (int64_t d : result.deletions) {
        del_mask[static_cast<size_t>(d)] = 1;
    }

    result.surviving.reserve(static_cast<size_t>(n));
    for (int idx = 0; idx < n; ++idx) {
        if (del_mask[static_cast<size_t>(idx)] == 0) {
            result.surviving.push_back(idx);
        }
    }

    const int num_gaps = std::max(0, static_cast<int>(result.surviving.size()) - 1);
    result.per_gap.assign(static_cast<size_t>(num_gaps), {});

    if (num_gaps > 0 && !insertions_raw.empty()) {
        // rank_of[i]  = rank of surviving token i (or -1 if deleted).
        // num_surv_le[i] = count of surviving tokens with index <= i.
        // Both sized n+1 so `after_y == n` (trailing insertions) is safe.
        std::vector<int> rank_of(static_cast<size_t>(n) + 1, -1);
        std::vector<int> num_surv_le(static_cast<size_t>(n) + 1, 0);
        {
            int cnt = 0;
            for (int idx = 0; idx < n; ++idx) {
                if (del_mask[static_cast<size_t>(idx)] == 0) {
                    rank_of[static_cast<size_t>(idx)] = cnt++;
                }
                num_surv_le[static_cast<size_t>(idx)] = cnt;
            }
            num_surv_le[static_cast<size_t>(n)] = cnt;
        }

        const size_t n_ins = insertions_raw.size();
        for (size_t k = 0; k < n_ins; ++k) {
            const int after_y = insertion_after[k];

            int gap_surv_idx;
            const int r = rank_of[static_cast<size_t>(after_y)];
            if (r >= 0) {
                gap_surv_idx = r - 1;  // gap BEFORE this position
            } else {
                // after_y points to a deleted position — last surviving
                // token before it is (count of survivors < after_y) - 1
                gap_surv_idx = (after_y > 0
                                    ? num_surv_le[static_cast<size_t>(after_y - 1)]
                                    : 0)
                               - 1;
            }

            if (gap_surv_idx >= 0 && gap_surv_idx < num_gaps) {
                result.per_gap[static_cast<size_t>(gap_surv_idx)].push_back(
                    insertions_raw[k]);
            }
        }
    }

    return result;
}

// ── Helpers: coercion + packed output ────────────────────────────────────

torch::Tensor coerce_int64_cpu(torch::Tensor t) {
    // Fast path: training already hands us CPU/int64/contiguous tensors, in
    // which case the three dispatches (contiguous → to → cpu) are pure
    // overhead — skip them and read the buffer directly.
    if (!(t.device().is_cpu() && t.scalar_type() == torch::kInt64 &&
          t.is_contiguous())) {
        t = t.contiguous().to(torch::kInt64).cpu();
    }
    return t;
}

// Helper: copy a std::vector into a new torch::Tensor
template <typename T>
torch::Tensor vector_to_tensor(const std::vector<T>& vec,
                               torch::ScalarType dtype) {
    auto t = torch::empty(
        {static_cast<int64_t>(vec.size())},
        torch::TensorOptions().dtype(dtype));
    if (!vec.empty()) {
        std::copy(vec.begin(), vec.end(), t.data_ptr<T>());
    }
    return t;
}

// Concatenate per-sample vectors into one flat tensor.
template <typename T>
torch::Tensor concat_to_tensor(const std::vector<std::vector<T>>& data,
                               torch::ScalarType dtype) {
    size_t total = 0;
    for (const auto& d : data) {
        total += d.size();
    }
    auto t = torch::empty(
        {static_cast<int64_t>(total)},
        torch::TensorOptions().dtype(dtype));
    T* dst = t.data_ptr<T>();
    for (const auto& d : data) {
        if (!d.empty()) {
            std::copy(d.begin(), d.end(), dst);
            dst += d.size();
        }
    }
    return t;
}

// Build [B+1] cumulative offsets for a per-sample length list.
torch::Tensor offsets_tensor(const std::vector<int64_t>& lengths) {
    std::vector<int64_t> off;
    off.reserve(lengths.size() + 1);
    int64_t acc = 0;
    off.push_back(0);
    for (int64_t l : lengths) {
        acc += l;
        off.push_back(acc);
    }
    return vector_to_tensor(off, torch::kInt64);
}

}  // anonymous namespace


std::tuple<torch::Tensor, std::vector<torch::Tensor>>
levenshtein_align(torch::Tensor y, torch::Tensor y_star) {
    y = coerce_int64_cpu(std::move(y));
    y_star = coerce_int64_cpu(std::move(y_star));

    const int n = static_cast<int>(y.size(0));
    const int m = static_cast<int>(y_star.size(0));

    RawAlign ra = align_one_raw(y.const_data_ptr<int64_t>(), n,
                                y_star.const_data_ptr<int64_t>(), m);

    auto deletions_tensor = vector_to_tensor(ra.deletions, torch::kInt64);
    std::vector<torch::Tensor> per_gap;
    per_gap.reserve(ra.per_gap.size());
    for (auto& gap_tokens : ra.per_gap) {
        per_gap.push_back(vector_to_tensor(gap_tokens, torch::kInt64));
    }
    return std::make_tuple(deletions_tensor, per_gap);
}


// ── Packed batch: deletion oracle ────────────────────────────────────────
//
// For each (y, y_star): mask[g] = 1 if g is deleted by the optimal alignment
// and g is not a boundary (positions 0 and n-1 are always kept). Mirrors
// expert.oracle_deletion.

std::tuple<torch::Tensor, torch::Tensor>
levenshtein_deletion_batch(std::vector<torch::Tensor> ys,
                           std::vector<torch::Tensor> ys_stars) {
    if (ys.size() != ys_stars.size()) {
        throw std::invalid_argument(
            "levenshtein_deletion_batch: list length mismatch");
    }
    const size_t B = ys.size();
    std::vector<std::vector<uint8_t>> masks(B);
    std::vector<int64_t> lengths(B);

    for (size_t i = 0; i < B; ++i) {
        torch::Tensor y = coerce_int64_cpu(std::move(ys[i]));
        torch::Tensor y_star = coerce_int64_cpu(std::move(ys_stars[i]));
        const int n = static_cast<int>(y.size(0));
        const int m = static_cast<int>(y_star.size(0));

        RawAlign ra = align_one_raw(y.const_data_ptr<int64_t>(), n,
                                    y_star.const_data_ptr<int64_t>(), m);

        std::vector<uint8_t>& mask = masks[i];
        mask.assign(static_cast<size_t>(n), 0);   // default: keep
        for (int64_t d : ra.deletions) {
            mask[static_cast<size_t>(d)] = 1;      // mark delete
        }
        if (n > 0) mask[0] = 0;                    // never delete BOS
        if (n > 1) mask[static_cast<size_t>(n - 1)] = 0;  // never delete EOS
        lengths[i] = n;
    }

    auto packed = concat_to_tensor(masks, torch::kUInt8);
    auto offsets = offsets_tensor(lengths);
    return std::make_tuple(packed, offsets);
}


// ── Packed batch: insertion oracle ───────────────────────────────────────
//
// For each (y, y_star) with y the FINAL roll-in sequence:
//   p_star[i]    length len(y)-1, zeros at gaps without insertions
//   t_star[i]    concat(per_gap[g][:max_placeholder]) in gap order
//   y_ins_plh[i] y with p_star[g] PLH tokens inserted after position g
// Mirrors expert.oracle_insertion + insert_placeholders.

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
levenshtein_insertion_batch(std::vector<torch::Tensor> ys,
                            std::vector<torch::Tensor> ys_stars,
                            int64_t max_placeholder,
                            int64_t plh_token_id) {
    if (ys.size() != ys_stars.size()) {
        throw std::invalid_argument(
            "levenshtein_insertion_batch: list length mismatch");
    }
    const size_t B = ys.size();
    std::vector<std::vector<int64_t>> p_stars(B), t_stars(B), y_ins_plhs(B);
    std::vector<int64_t> p_lens(B), t_lens(B), plh_lens(B);

    for (size_t i = 0; i < B; ++i) {
        torch::Tensor y = coerce_int64_cpu(std::move(ys[i]));
        torch::Tensor y_star = coerce_int64_cpu(std::move(ys_stars[i]));
        const int n = static_cast<int>(y.size(0));
        if (n < 1) {
            throw std::invalid_argument(
                "levenshtein_insertion_batch: y must be non-empty");
        }
        const int m = static_cast<int>(y_star.size(0));
        const int64_t* y_ptr = y.const_data_ptr<int64_t>();

        RawAlign ra = align_one_raw(y_ptr, n, y_star.const_data_ptr<int64_t>(), m);

        std::vector<int64_t>& p_star = p_stars[i];
        std::vector<int64_t>& t_star = t_stars[i];
        std::vector<int64_t>& plh_out = y_ins_plhs[i];

        // p_star: placeholder count per gap, anchored at surviving[g].
        // surviving[num_gaps-1] is the second-to-last survivor, so the index
        // stays within [0, n-2] (last survivor = right boundary, never anchor).
        p_star.assign(static_cast<size_t>(n - 1), 0);
        const size_t num_gaps = ra.per_gap.size();
        for (size_t g = 0; g < num_gaps; ++g) {
            const size_t cnt = ra.per_gap[g].size();
            const int64_t capped = static_cast<int64_t>(
                std::min<size_t>(cnt, static_cast<size_t>(max_placeholder)));
            const int anchor = ra.surviving[g];
            p_star[static_cast<size_t>(anchor)] = capped;
            for (size_t k = 0; k < static_cast<size_t>(capped); ++k) {
                t_star.push_back(ra.per_gap[g][k]);
            }
        }

        // y_ins_plh: y[pos], then p_star[pos] PLH tokens, then final token.
        plh_out.reserve(static_cast<size_t>(n) + t_star.size());
        for (int pos = 0; pos < n - 1; ++pos) {
            plh_out.push_back(y_ptr[pos]);
            const int64_t cnt = p_star[static_cast<size_t>(pos)];
            for (int64_t k = 0; k < cnt; ++k) {
                plh_out.push_back(plh_token_id);
            }
        }
        plh_out.push_back(y_ptr[n - 1]);

        p_lens[i] = n - 1;
        t_lens[i] = static_cast<int64_t>(t_star.size());
        plh_lens[i] = static_cast<int64_t>(plh_out.size());
    }

    auto p_packed = concat_to_tensor(p_stars, torch::kInt64);
    auto p_offsets = offsets_tensor(p_lens);
    auto t_packed = concat_to_tensor(t_stars, torch::kInt64);
    auto t_offsets = offsets_tensor(t_lens);
    auto plh_packed = concat_to_tensor(y_ins_plhs, torch::kInt64);
    auto plh_offsets = offsets_tensor(plh_lens);
    return std::make_tuple(p_packed, p_offsets,
                           t_packed, t_offsets,
                           plh_packed, plh_offsets);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "C++ Levenshtein alignment for the Levenshtein Transformer";
    m.def("levenshtein_align", &levenshtein_align,
          "Compute optimal edit alignment (insert+delete only, no substitution).\n\n"
          "Args:\n"
          "    y:      1-D int64 tensor — current token sequence (includes BOS/EOS)\n"
          "    y_star: 1-D int64 tensor — target token sequence (includes BOS/EOS)\n\n"
          "Returns:\n"
          "    (deletions, insertions) tuple\n"
          "    deletions:  1-D int64 tensor — indices in y to delete\n"
          "    insertions: list of 1-D int64 tensors — per-gap token lists",
          pybind11::arg("y"), pybind11::arg("y_star"));

    m.def("levenshtein_deletion_batch", &levenshtein_deletion_batch,
          "Batched deletion oracle: packed boundary-protected deletion masks.\n\n"
          "Args:\n"
          "    ys:       list of 1-D int64 CPU tensors (current sequences)\n"
          "    ys_stars: list of 1-D int64 CPU tensors (targets, same length)\n\n"
          "Returns:\n"
          "    (mask_packed, offsets) tuple\n"
          "    mask_packed: uint8 tensor — per-sample concatenated delete flags\n"
          "    offsets:     int64 tensor [B+1] — per-sample boundaries into mask_packed",
          pybind11::arg("ys"), pybind11::arg("ys_stars"));

    m.def("levenshtein_insertion_batch", &levenshtein_insertion_batch,
          "Batched insertion oracle: packed placeholder counts, capped tokens,\n"
          "and PLH-interleaved roll-in.\n\n"
          "Args:\n"
          "    ys:              list of 1-D int64 CPU tensors (final roll-ins)\n"
          "    ys_stars:        list of 1-D int64 CPU tensors (targets)\n"
          "    max_placeholder: per-gap placeholder cap\n"
          "    plh_token_id:    <PLH> token id\n\n"
          "Returns:\n"
          "    (p_star_packed, p_star_offsets, t_star_packed, t_star_offsets,\n"
          "     y_ins_plh_packed, y_ins_plh_offsets) tuple — each pair is a\n"
          "    flat concatenation plus a [B+1] offsets tensor.",
          pybind11::arg("ys"), pybind11::arg("ys_stars"),
          pybind11::arg("max_placeholder"), pybind11::arg("plh_token_id"));
}
