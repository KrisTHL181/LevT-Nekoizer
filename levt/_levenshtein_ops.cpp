/**
 * C++ Levenshtein alignment — insert + delete only (no substitution).
 *
 * Uses LibTorch tensors directly (no Python list serialization overhead).
 *
 * DP recurrence: substitution cost = 2 (i.e. delete + insert).
 * This is equivalent to disallowing substitution and allowing only
 * insertion (cost 1) + deletion (cost 1) + keep-if-equal (cost 0).
 */

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace {

// ── Flat-array DP tables (1-D row-major, better cache locality) ──────────

struct DPResult {
    std::vector<int> d;        // (n+1) × (m+1) flattened, row-major
    std::vector<uint8_t> back; // (n+1) × (m+1) flattened, row-major
    int n, m;

    int  d_at(int i, int j) const { return d[i * (m + 1) + j]; }
    int& d_at(int i, int j)       { return d[i * (m + 1) + j]; }
    uint8_t  back_at(int i, int j) const { return back[i * (m + 1) + j]; }
    uint8_t& back_at(int i, int j)       { return back[i * (m + 1) + j]; }
};

DPResult edit_distance2_with_dp(const int64_t* x, int n,
                                const int64_t* y, int m) {
    DPResult result;
    result.n = n;
    result.m = m;
    const auto size = static_cast<size_t>((n + 1) * (m + 1));
    result.d.resize(size);
    result.back.resize(size);

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
        for (int j = 1; j <= m; ++j) {
            int best_val = std::numeric_limits<int>::max();
            uint8_t best_op = 0;

            // Delete: cost 1
            int cand = result.d_at(i - 1, j) + 1;
            if (cand < best_val) {
                best_val = cand;
                best_op = 1;
            }

            // Insert: cost 1
            cand = result.d_at(i, j - 1) + 1;
            if (cand < best_val) {
                best_val = cand;
                best_op = 2;
            }

            // Match: cost 0 if equal, can't substitute (cost would be 2)
            if (xi == y[j - 1]) {
                cand = result.d_at(i - 1, j - 1);
                if (cand < best_val) {
                    best_val = cand;
                    best_op = 0;
                }
            }

            result.d_at(i, j) = best_val;
            result.back_at(i, j) = best_op;
        }
    }

    return result;
}

// ── Helper: copy a std::vector into a new torch::Tensor ────────────────

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

}  // anonymous namespace


std::tuple<torch::Tensor, std::vector<torch::Tensor>>
levenshtein_align(torch::Tensor y, torch::Tensor y_star) {
    // ── Ensure CPU, int64, contiguous ──────────────────────────────────
    y = y.contiguous().to(torch::kInt64).cpu();
    y_star = y_star.contiguous().to(torch::kInt64).cpu();

    const int n = static_cast<int>(y.size(0));
    const int m = static_cast<int>(y_star.size(0));

    const int64_t* y_ptr = y.const_data_ptr<int64_t>();
    const int64_t* ys_ptr = y_star.const_data_ptr<int64_t>();

    // ── DP ─────────────────────────────────────────────────────────────
    DPResult dp = edit_distance2_with_dp(y_ptr, n, ys_ptr, m);

    // ── Pass 1: collect deletions (backtrack) ──────────────────────────
    std::vector<int64_t> deletions;
    {
        int i = n, j = m;
        while (i > 0 || j > 0) {
            uint8_t op = dp.back_at(i, j);
            if (op == 0) {        // match
                --i; --j;
            } else if (op == 1) { // delete
                --i;
                deletions.push_back(static_cast<int64_t>(i));
            } else {              // insert
                --j;
            }
        }
        std::reverse(deletions.begin(), deletions.end());
    }

    // ── Pass 2: collect insertions (token, after_y_index) ──────────────
    std::vector<int64_t> insertions_raw;
    std::vector<int> insertion_after;

    {
        int i = n, j = m;
        int current_after = n;
        while (i > 0 || j > 0) {
            uint8_t op = dp.back_at(i, j);
            if (op == 0) {        // match
                --i; --j;
                current_after = i;
            } else if (op == 1) { // delete
                --i;
            } else {              // insert
                --j;
                insertions_raw.push_back(ys_ptr[j]);
                insertion_after.push_back(current_after);
            }
        }
        std::reverse(insertions_raw.begin(), insertions_raw.end());
        std::reverse(insertion_after.begin(), insertion_after.end());
    }

    // ── Build surviving-set and per-gap insertion lists ────────────────
    // Boolean mask for O(1) deletion lookup (replaces std::set<int>).
    std::vector<uint8_t> del_mask(static_cast<size_t>(n), 0);
    for (int64_t d : deletions) {
        del_mask[static_cast<size_t>(d)] = 1;
    }

    std::vector<int> surviving;
    surviving.reserve(static_cast<size_t>(n));
    for (int idx = 0; idx < n; ++idx) {
        if (del_mask[static_cast<size_t>(idx)] == 0) {
            surviving.push_back(idx);
        }
    }

    const int num_gaps = std::max(0, static_cast<int>(surviving.size()) - 1);
    std::vector<std::vector<int64_t>> per_gap_vecs(num_gaps);

    if (num_gaps > 0 && !insertions_raw.empty()) {
        std::unordered_map<int, int> surv_rank;
        surv_rank.reserve(surviving.size());
        for (int si = 0; si < static_cast<int>(surviving.size()); ++si) {
            surv_rank[surviving[si]] = si;
        }

        const size_t n_ins = insertions_raw.size();
        for (size_t k = 0; k < n_ins; ++k) {
            int64_t tok = insertions_raw[k];
            int after_y = insertion_after[k];

            int gap_surv_idx = -1;
            auto it = surv_rank.find(after_y);
            if (it != surv_rank.end()) {
                gap_surv_idx = it->second - 1;  // gap BEFORE this position
            } else {
                // after_y points to a deleted position — find last surviving
                // token before it
                const int n_surv = static_cast<int>(surviving.size());
                for (int si = 0; si < n_surv; ++si) {
                    if (surviving[si] < after_y) {
                        gap_surv_idx = si;
                    } else {
                        break;
                    }
                }
            }

            if (gap_surv_idx >= 0 && gap_surv_idx < num_gaps) {
                per_gap_vecs[gap_surv_idx].push_back(tok);
            }
        }
    }

    // ── Convert to tensors ─────────────────────────────────────────────
    auto deletions_tensor = vector_to_tensor(deletions, torch::kInt64);

    std::vector<torch::Tensor> per_gap;
    per_gap.reserve(static_cast<size_t>(num_gaps));
    for (auto& gap_tokens : per_gap_vecs) {
        per_gap.push_back(
            vector_to_tensor(gap_tokens, torch::kInt64));
    }

    return std::make_tuple(deletions_tensor, per_gap);
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
}
