#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "batched_scipy_lap.h"
#include "rectangular_lsap.h"
#include <vector>

namespace py = pybind11;

// Wrapper for double precision batched solver
py::array_t<int64_t> py_solve_batched_lap_double(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_matrices,
    bool maximize,
    py::object num_valid_obj,
    int64_t unassigned_value,
    bool use_openmp
) {
    auto buf = cost_matrices.request();

    if (buf.ndim != 3) {
        throw std::runtime_error("Cost matrices must be a 3D array (batch_size, nr, nc)");
    }

    int64_t batch_size = buf.shape[0];
    int64_t nr = buf.shape[1];
    int64_t nc = buf.shape[2];

    const double* cost_ptr = static_cast<const double*>(buf.ptr);

    // Handle num_valid parameter (can be None, scalar, or array)
    std::vector<int64_t> num_valid_vec;
    const int64_t* num_valid_ptr = nullptr;

    if (!num_valid_obj.is_none()) {
        if (py::isinstance<py::int_>(num_valid_obj)) {
            // Scalar value - replicate for all batches
            int64_t val = num_valid_obj.cast<int64_t>();
            num_valid_vec.resize(batch_size, val);
            num_valid_ptr = num_valid_vec.data();
        } else {
            // Array of values
            auto num_valid_array = num_valid_obj.cast<py::array_t<int64_t>>();
            auto nv_buf = num_valid_array.request();
            if (nv_buf.ndim != 1 || nv_buf.shape[0] != batch_size) {
                throw std::runtime_error("num_valid must be scalar or 1D array of length batch_size");
            }
            num_valid_ptr = static_cast<const int64_t*>(nv_buf.ptr);
        }
    }

    // Allocate output array
    auto result = py::array_t<int64_t>({batch_size, nr});
    auto result_buf = result.request();
    int64_t* result_ptr = static_cast<int64_t*>(result_buf.ptr);

    // Call the C++ function
    int status = batched_scipy::solve_batched_lap_double(
        batch_size, nr, nc,
        cost_ptr,
        maximize,
        result_ptr,
        num_valid_ptr,
        unassigned_value,
        use_openmp
    );

    if (status != 0) {
        throw std::runtime_error("Batched LAP solver failed with status: " + std::to_string(status));
    }

    return result;
}

// Wrapper for single precision batched solver
py::array_t<int64_t> py_solve_batched_lap_float(
    py::array_t<float, py::array::c_style | py::array::forcecast> cost_matrices,
    bool maximize,
    py::object num_valid_obj,
    int64_t unassigned_value,
    bool use_openmp
) {
    auto buf = cost_matrices.request();

    if (buf.ndim != 3) {
        throw std::runtime_error("Cost matrices must be a 3D array (batch_size, nr, nc)");
    }

    int64_t batch_size = buf.shape[0];
    int64_t nr = buf.shape[1];
    int64_t nc = buf.shape[2];

    const float* cost_ptr = static_cast<const float*>(buf.ptr);

    // Handle num_valid parameter (can be None, scalar, or array)
    std::vector<int64_t> num_valid_vec;
    const int64_t* num_valid_ptr = nullptr;

    if (!num_valid_obj.is_none()) {
        if (py::isinstance<py::int_>(num_valid_obj)) {
            // Scalar value - replicate for all batches
            int64_t val = num_valid_obj.cast<int64_t>();
            num_valid_vec.resize(batch_size, val);
            num_valid_ptr = num_valid_vec.data();
        } else {
            // Array of values
            auto num_valid_array = num_valid_obj.cast<py::array_t<int64_t>>();
            auto nv_buf = num_valid_array.request();
            if (nv_buf.ndim != 1 || nv_buf.shape[0] != batch_size) {
                throw std::runtime_error("num_valid must be scalar or 1D array of length batch_size");
            }
            num_valid_ptr = static_cast<const int64_t*>(nv_buf.ptr);
        }
    }

    // Allocate output array
    auto result = py::array_t<int64_t>({batch_size, nr});
    auto result_buf = result.request();
    int64_t* result_ptr = static_cast<int64_t*>(result_buf.ptr);

    // Call the C++ function
    int status = batched_scipy::solve_batched_lap_float(
        batch_size, nr, nc,
        cost_ptr,
        maximize,
        result_ptr,
        num_valid_ptr,
        unassigned_value,
        use_openmp
    );

    if (status != 0) {
        throw std::runtime_error("Batched LAP solver failed with status: " + std::to_string(status));
    }

    return result;
}

PYBIND11_MODULE(_batched_scipy_lap, m) {
    m.doc() = "Batched scipy LAP solver with OpenMP parallelization";

    m.def("solve_batched_lap_double", &py_solve_batched_lap_double,
          py::arg("cost_matrices"),
          py::arg("maximize") = false,
          py::arg("num_valid") = py::none(),
          py::arg("unassigned_value") = -1,
          py::arg("use_openmp") = true,
          "Solve batched LAP with double precision");

    m.def("solve_batched_lap_float", &py_solve_batched_lap_float,
          py::arg("cost_matrices"),
          py::arg("maximize") = false,
          py::arg("num_valid") = py::none(),
          py::arg("unassigned_value") = -1,
          py::arg("use_openmp") = true,
          "Solve batched LAP with single precision");

    // Feature detection
    #ifdef LAP_OPENMP
    m.attr("HAS_OPENMP") = true;
    #else
    m.attr("HAS_OPENMP") = false;
    #endif
}
