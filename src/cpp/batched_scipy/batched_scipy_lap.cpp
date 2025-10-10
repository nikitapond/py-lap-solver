#include "batched_scipy_lap.h"
#include "rectangular_lsap.h"
#include <vector>
#include <algorithm>
#include <cstring>

#ifdef LAP_OPENMP
#include <omp.h>
#endif

namespace batched_scipy {

template<typename T>
int solve_batched_lap_impl(
    int64_t batch_size,
    int64_t nr,
    int64_t nc,
    const T* cost_matrices,
    bool maximize,
    int64_t* row_assignments,
    const int64_t* num_valid,
    int64_t unassigned_value,
    bool use_openmp
) {
    // Validate inputs
    if (batch_size <= 0 || nr <= 0 || nc <= 0) {
        return -1;
    }
    if (cost_matrices == nullptr || row_assignments == nullptr) {
        return -2;
    }

    int overall_status = 0;

    // Decide whether to use OpenMP at runtime
    bool run_parallel = false;
#ifdef LAP_OPENMP
    run_parallel = use_openmp;
#endif

    // Process each problem in the batch
    if (run_parallel) {
#ifdef LAP_OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int64_t b = 0; b < batch_size; b++) {
            // Get the dimensions for this problem
            // num_valid only limits rows, keep all columns
            int64_t nr_valid = num_valid ? num_valid[b] : nr;
            int64_t nc_valid = nc;

        // Clamp to actual dimensions
        nr_valid = std::min(nr_valid, nr);
        nc_valid = std::min(nc_valid, nc);

        // Get pointer to this batch's cost matrix
        const T* cost_matrix = cost_matrices + b * nr * nc;

        // Convert to double and create a contiguous copy for the valid region
        std::vector<double> cost_double(nr_valid * nc_valid);
        for (int64_t i = 0; i < nr_valid; i++) {
            for (int64_t j = 0; j < nc_valid; j++) {
                cost_double[i * nc_valid + j] = static_cast<double>(cost_matrix[i * nc + j]);
            }
        }

        // Allocate output arrays for scipy solver
        // The solver returns min(nr, nc) assignments
        int64_t num_assignments = std::min(nr_valid, nc_valid);
        std::vector<int64_t> row_ind(num_assignments);
        std::vector<int64_t> col_ind(num_assignments);

        // Call the scipy solver
        int status = solve_rectangular_linear_sum_assignment(
            nr_valid, nc_valid,
            cost_double.data(),
            maximize,
            row_ind.data(),
            col_ind.data()
        );

        // Get pointer to output for this batch
        int64_t* output = row_assignments + b * nr;

        if (status == 0) {
            // Initialize all assignments to unassigned_value
            std::fill(output, output + nr, unassigned_value);

            // Fill in the valid assignments
            // row_ind and col_ind are arrays of length min(nr_valid, nc_valid)
            for (int64_t i = 0; i < num_assignments; i++) {
                if (row_ind[i] >= 0 && row_ind[i] < nr) {
                    output[row_ind[i]] = col_ind[i];
                }
            }
        } else {
            // On error, fill with unassigned values
            std::fill(output, output + nr, unassigned_value);
#ifdef LAP_OPENMP
            if (use_openmp) {
                #pragma omp critical
                {
                    if (overall_status == 0) {
                        overall_status = status;
                    }
                }
            } else {
                if (overall_status == 0) {
                    overall_status = status;
                }
            }
#else
            if (overall_status == 0) {
                overall_status = status;
            }
#endif
            }  // end if/else
        }  // end for loop (parallel path)
    } else {
        for (int64_t b = 0; b < batch_size; b++) {
            // Get the dimensions for this problem
            // num_valid only limits rows, keep all columns
            int64_t nr_valid = num_valid ? num_valid[b] : nr;
            int64_t nc_valid = nc;

            // Clamp to actual dimensions
            nr_valid = std::min(nr_valid, nr);
            nc_valid = std::min(nc_valid, nc);

            // Get pointer to this batch's cost matrix
            const T* cost_matrix = cost_matrices + b * nr * nc;

            // Convert to double and create a contiguous copy for the valid region
            std::vector<double> cost_double(nr_valid * nc_valid);
            for (int64_t i = 0; i < nr_valid; i++) {
                for (int64_t j = 0; j < nc_valid; j++) {
                    cost_double[i * nc_valid + j] = static_cast<double>(cost_matrix[i * nc + j]);
                }
            }

            // Allocate output arrays for scipy solver
            // The solver returns min(nr, nc) assignments
            int64_t num_assignments = std::min(nr_valid, nc_valid);
            std::vector<int64_t> row_ind(num_assignments);
            std::vector<int64_t> col_ind(num_assignments);

            // Call the scipy solver
            int status = solve_rectangular_linear_sum_assignment(
                nr_valid, nc_valid,
                cost_double.data(),
                maximize,
                row_ind.data(),
                col_ind.data()
            );

            // Get pointer to output for this batch
            int64_t* output = row_assignments + b * nr;

            if (status == 0) {
                // Initialize all assignments to unassigned_value
                std::fill(output, output + nr, unassigned_value);

                // Fill in the valid assignments
                // row_ind and col_ind are arrays of length min(nr_valid, nc_valid)
                for (int64_t i = 0; i < num_assignments; i++) {
                    if (row_ind[i] >= 0 && row_ind[i] < nr) {
                        output[row_ind[i]] = col_ind[i];
                    }
                }
            } else {
                // On error, fill with unassigned values
                std::fill(output, output + nr, unassigned_value);
                if (overall_status == 0) {
                    overall_status = status;
                }
            }
        }  // end for loop (sequential path)
    }  // end if (run_parallel)

    return overall_status;
}

int solve_batched_lap_double(
    int64_t batch_size,
    int64_t nr,
    int64_t nc,
    const double* cost_matrices,
    bool maximize,
    int64_t* row_assignments,
    const int64_t* num_valid,
    int64_t unassigned_value,
    bool use_openmp
) {
    return solve_batched_lap_impl<double>(
        batch_size, nr, nc, cost_matrices, maximize,
        row_assignments, num_valid, unassigned_value, use_openmp
    );
}

int solve_batched_lap_float(
    int64_t batch_size,
    int64_t nr,
    int64_t nc,
    const float* cost_matrices,
    bool maximize,
    int64_t* row_assignments,
    const int64_t* num_valid,
    int64_t unassigned_value,
    bool use_openmp
) {
    return solve_batched_lap_impl<float>(
        batch_size, nr, nc, cost_matrices, maximize,
        row_assignments, num_valid, unassigned_value, use_openmp
    );
}

} // namespace batched_scipy
