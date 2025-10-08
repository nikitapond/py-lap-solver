#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>
#include "lap.h"

namespace py = pybind11;

// Wrapper function for single LAP problem with float precision
py::array_t<int32_t> solve_lap_float(
    py::array_t<float, py::array::c_style | py::array::forcecast> cost_matrix,
    int num_valid = -1
) {
    auto buf = cost_matrix.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Cost matrix must be a 2D array");
    }

    int64_t n_rows = buf.shape[0];
    int64_t n_cols = buf.shape[1];

    // Use num_valid if provided, otherwise use matrix dimensions
    int dim_rows = (num_valid > 0) ? num_valid : n_rows;
    int dim_cols = (num_valid > 0) ? num_valid : n_cols;
    int dim = std::min(dim_rows, dim_cols);

    float* cost_ptr = const_cast<float*>(static_cast<const float*>(buf.ptr));

    // Create TableCost wrapper (references existing array, doesn't copy)
    lap::TableCost<float> costfunc(n_rows, n_cols, cost_ptr);

    // Create iterator (needs reference to costfunc)
    lap::DirectIterator<float, lap::TableCost<float>> iterator(costfunc);

    // Allocate output array for row assignments
    std::vector<int> rowsol(dim);

    // Solve the LAP (rectangular version if dims differ)
    if (dim_rows == dim_cols) {
        lap::solve<float>(dim, costfunc, iterator, rowsol.data(), true);
    } else {
        lap::solve<float>(dim_rows, dim_cols, costfunc, iterator, rowsol.data(), true);
    }

    // Convert to output format (row_to_col mapping for all rows)
    auto result = py::array_t<int32_t>(n_rows);
    auto result_buf = result.request();
    int32_t* result_ptr = static_cast<int32_t*>(result_buf.ptr);

    // Initialize all to -1 (unassigned)
    std::fill(result_ptr, result_ptr + n_rows, -1);

    // Fill in the assignments from the solver
    for (int i = 0; i < dim; i++) {
        result_ptr[i] = rowsol[i];
    }

    return result;
}

// Wrapper function for single LAP problem with double precision
py::array_t<int32_t> solve_lap_double(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_matrix,
    int num_valid = -1
) {
    auto buf = cost_matrix.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Cost matrix must be a 2D array");
    }

    int64_t n_rows = buf.shape[0];
    int64_t n_cols = buf.shape[1];

    // Use num_valid if provided, otherwise use matrix dimensions
    int dim_rows = (num_valid > 0) ? num_valid : n_rows;
    int dim_cols = (num_valid > 0) ? num_valid : n_cols;
    int dim = std::min(dim_rows, dim_cols);

    double* cost_ptr = const_cast<double*>(static_cast<const double*>(buf.ptr));

    // Create TableCost wrapper (references existing array, doesn't copy)
    lap::TableCost<double> costfunc(n_rows, n_cols, cost_ptr);

    // Create iterator (needs reference to costfunc)
    lap::DirectIterator<double, lap::TableCost<double>> iterator(costfunc);

    // Allocate output array for row assignments
    std::vector<int> rowsol(dim);

    // Solve the LAP (rectangular version if dims differ)
    if (dim_rows == dim_cols) {
        lap::solve<double>(dim, costfunc, iterator, rowsol.data(), true);
    } else {
        lap::solve<double>(dim_rows, dim_cols, costfunc, iterator, rowsol.data(), true);
    }

    // Convert to output format (row_to_col mapping for all rows)
    auto result = py::array_t<int32_t>(n_rows);
    auto result_buf = result.request();
    int32_t* result_ptr = static_cast<int32_t*>(result_buf.ptr);

    // Initialize all to -1 (unassigned)
    std::fill(result_ptr, result_ptr + n_rows, -1);

    // Fill in the assignments from the solver
    for (int i = 0; i < dim; i++) {
        result_ptr[i] = rowsol[i];
    }

    return result;
}

// Note: OpenMP parallel versions are not yet implemented
// The LAP1015 OpenMP API requires additional Worksharing setup
// For now, use the sequential version

PYBIND11_MODULE(_lap1015, m) {
    m.doc() = "LAP1015 solver - Algorithm 1015 for Linear Assignment Problem";

    // Sequential versions
    m.def("solve_lap_float", &solve_lap_float,
          py::arg("cost_matrix"),
          py::arg("num_valid") = -1,
          "Solve LAP with single precision (float32)");

    m.def("solve_lap_double", &solve_lap_double,
          py::arg("cost_matrix"),
          py::arg("num_valid") = -1,
          "Solve LAP with double precision (float64)");

    // OpenMP versions not yet exposed (require Worksharing setup)

    // Feature detection flags
    #ifdef LAP_OPENMP
    m.attr("HAS_OPENMP") = true;
    #else
    m.attr("HAS_OPENMP") = false;
    #endif

    #ifdef LAP_CUDA
    m.attr("HAS_CUDA") = true;
    #else
    m.attr("HAS_CUDA") = false;
    #endif
}
