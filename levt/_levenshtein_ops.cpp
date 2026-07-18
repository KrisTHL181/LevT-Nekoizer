/**
 * C++ Levenshtein alignment — insert + delete only (no substitution).
 *
 * Ported from fairseq's libnat/edit_dist.cpp and adapted to return the
 * same (deletions, per_gap) format as the pure-Python levt.expert path.
 *
 * DP recurrence: substitution cost = 2 (i.e. delete + insert).
 * This is equivalent to disallowing substitution and allowing only
 * insertion (cost 1) + deletion (cost 1) + keep-if-equal (cost 0).
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <unordered_map>
#include <vector>

namespace {

struct DPResult {
    std::vector<std::vector<int>> d;
    std::vector<std::vector<uint8_t>> back;
};

DPResult edit_distance2_with_dp(const std::vector<int64_t>& x,
                                const std::vector<int64_t>& y) {
    const int n = static_cast<int>(x.size());
    const int m = static_cast<int>(y.size());

    DPResult result;
    result.d.resize(n + 1, std::vector<int>(m + 1));
    result.back.resize(n + 1, std::vector<uint8_t>(m + 1));

    // Initialize boundaries
    for (int i = 0; i <= n; i++) {
        result.d[i][0] = i;
        result.back[i][0] = 1;  // delete
    }
    for (int j = 0; j <= m; j++) {
        result.d[0][j] = j;
        result.back[0][j] = 2;  // insert
    }

    // DP fill
    for (int i = 1; i <= n; i++) {
        const int64_t xi = x[i - 1];
        for (int j = 1; j <= m; j++) {
            int best_val = std::numeric_limits<int>::max();
            uint8_t best_op = 0;

            // Delete: cost 1
            int cand = result.d[i - 1][j] + 1;
            if (cand < best_val) {
                best_val = cand;
                best_op = 1;
            }

            // Insert: cost 1
            cand = result.d[i][j - 1] + 1;
            if (cand < best_val) {
                best_val = cand;
                best_op = 2;
            }

            // Match: cost 0 if equal, can't substitute (cost would be 2)
            if (xi == y[j - 1]) {
                cand = result.d[i - 1][j - 1];
                if (cand < best_val) {
                    best_val = cand;
                    best_op = 0;
                }
            }

            result.d[i][j] = best_val;
            result.back[i][j] = best_op;
        }
    }

    return result;
}

}  // anonymous namespace


std::pair<std::vector<int64_t>, std::vector<std::vector<int64_t>>>
levenshtein_align(const std::vector<int64_t>& y,
                  const std::vector<int64_t>& y_star) {
    const int n = static_cast<int>(y.size());
    const int m = static_cast<int>(y_star.size());

    DPResult dp = edit_distance2_with_dp(y, y_star);
    const auto& back = dp.back;

    // ── Pass 1: collect deletions ──────────────────────────────────
    std::vector<int> deletions;
    {
        int i = n, j = m;
        while (i > 0 || j > 0) {
            uint8_t op = back[i][j];
            if (op == 0) {        // match
                i--; j--;
            } else if (op == 1) { // delete
                i--;
                deletions.push_back(i);
            } else {              // insert
                j--;
            }
        }
        std::reverse(deletions.begin(), deletions.end());
    }

    // ── Pass 2: collect insertions (token, after_y_index) ──────────
    std::vector<int64_t> insertions_raw;
    std::vector<int> insertion_after;

    {
        int i = n, j = m;
        int current_after = n;
        while (i > 0 || j > 0) {
            uint8_t op = back[i][j];
            if (op == 0) {        // match
                i--; j--;
                current_after = i;
            } else if (op == 1) { // delete
                i--;
            } else {              // insert
                j--;
                insertions_raw.push_back(y_star[j]);
                insertion_after.push_back(current_after);
            }
        }
        std::reverse(insertions_raw.begin(), insertions_raw.end());
        std::reverse(insertion_after.begin(), insertion_after.end());
    }

    // ── Build surviving-set and per-gap insertion lists ────────────
    std::set<int> del_set(deletions.begin(), deletions.end());
    std::vector<int> surviving;
    for (int idx = 0; idx < n; idx++) {
        if (del_set.find(idx) == del_set.end()) {
            surviving.push_back(idx);
        }
    }

    const int num_gaps = std::max(0, static_cast<int>(surviving.size()) - 1);
    std::vector<std::vector<int64_t>> per_gap(num_gaps);

    if (num_gaps > 0 && !insertions_raw.empty()) {
        std::unordered_map<int, int> surv_rank;
        for (int si = 0; si < static_cast<int>(surviving.size()); si++) {
            surv_rank[surviving[si]] = si;
        }

        for (size_t k = 0; k < insertions_raw.size(); k++) {
            int64_t tok = insertions_raw[k];
            int after_y = insertion_after[k];

            int gap_surv_idx = -1;
            auto it = surv_rank.find(after_y);
            if (it != surv_rank.end()) {
                gap_surv_idx = it->second - 1;  // gap BEFORE this position
            } else {
                // after_y points to a deleted position — find last surviving
                // token before it
                for (int si = 0; si < static_cast<int>(surviving.size()); si++) {
                    if (surviving[si] < after_y) {
                        gap_surv_idx = si;
                    } else {
                        break;
                    }
                }
            }

            if (gap_surv_idx >= 0 && gap_surv_idx < num_gaps) {
                per_gap[gap_surv_idx].push_back(tok);
            }
        }
    }

    std::vector<int64_t> deletions_i64(deletions.begin(), deletions.end());
    return {deletions_i64, per_gap};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "C++ Levenshtein alignment for the Levenshtein Transformer";
    m.def("levenshtein_align", &levenshtein_align,
          "Compute optimal edit alignment (insert+delete only, no substitution).\n\n"
          "Args:\n"
          "    y:      current token sequence (list of int)\n"
          "    y_star: target token sequence (list of int)\n\n"
          "Returns:\n"
          "    (deletions, insertions) tuple\n"
          "    deletions:  indices in y to delete\n"
          "    insertions: per-gap token lists for surviving positions",
          pybind11::arg("y"), pybind11::arg("y_star"));
}
