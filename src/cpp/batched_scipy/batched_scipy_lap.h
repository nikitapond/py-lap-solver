#ifndef BATCHED_SCIPY_LAP_H
#define BATCHED_SCIPY_LAP_H

#include <cstdint>
#include <vector>

namespace batched_scipy {

/**
 * Solve a batch of linear assignment problems using the scipy algorithm
 * with optional OpenMP parallelization across the batch dimension.
 *
 * @param batch_size Number of problems in the batch
 * @param nr Number of rows in each cost matrix
 * @param nc Number of columns in each cost matrix
 * @param cost_matrices Flattened array of cost matrices (shape: batch_size * nr * nc)
 * @param maximize If true, maximize instead of minimize
 * @param row_assignments Output array for row assignments (shape: batch_size * nr)
 *                        Each element contains the column index assigned to that row
 * @param num_valid Optional array of valid row dimensions for each matrix (can be nullptr)
 *                  If provided, only the first num_valid[b] rows are used for matrix b
 *                  All columns are always used
 * @param unassigned_value Value to use for unassigned rows (default: -1)
 * @param use_openmp Whether to use OpenMP parallelization (default: true)
 * @return 0 on success, negative on error
 */
int solve_batched_lap_double(
    int64_t batch_size,
    int64_t nr,
    int64_t nc,
    const double* cost_matrices,
    bool maximize,
    int64_t* row_assignments,
    const int64_t* num_valid = nullptr,
    int64_t unassigned_value = -1,
    bool use_openmp = true
);

int solve_batched_lap_float(
    int64_t batch_size,
    int64_t nr,
    int64_t nc,
    const float* cost_matrices,
    bool maximize,
    int64_t* row_assignments,
    const int64_t* num_valid = nullptr,
    int64_t unassigned_value = -1,
    bool use_openmp = true
);

} // namespace batched_scipy

#endif // BATCHED_SCIPY_LAP_H
