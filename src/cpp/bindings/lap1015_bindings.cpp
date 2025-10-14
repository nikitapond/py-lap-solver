#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "lap.h"
#include <chrono>

namespace py = pybind11;

#ifdef LAP_OPENMP
template <class SC, class TC, class CF, class TP>
void solveTableOMP(TP &start_time, int N1, int N2, CF &get_cost, int *rowsol, bool eps, bool sequential = false)
{
	lap::omp::SimpleCostFunction<TC, CF> costFunction(get_cost, sequential);
	lap::omp::Worksharing ws(N2, 8);
	lap::omp::TableCost<TC> costMatrix(N1, N2, costFunction, ws);
	lap::omp::DirectIterator<SC, TC, lap::omp::TableCost<TC>> iterator(N1, N2, costMatrix, ws);
	lap::omp::solve<SC>(N1, N2, costMatrix, iterator, rowsol, eps);
}
#endif

template <class SC, class TC, class CF, class TP>
void solveTable(TP &start_time, int N1, int N2, CF &get_cost, int *rowsol, bool eps)
{
	lap::SimpleCostFunction<TC, CF> costFunction(get_cost);
	lap::TableCost<TC> costMatrix(N1, N2, costFunction);
	lap::DirectIterator<SC, TC, lap::TableCost<TC>> iterator(N1, N2, costMatrix);
	lap::solve<SC>(N1, N2, costMatrix, iterator, rowsol, eps);
}

py::array_t<int32_t> solve_lap_float(
    py::array_t<float> cost_matrix,
    int num_valid = -1,
    bool use_openmp = false,
    bool use_epsilon = true
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto C = cost_matrix.mutable_unchecked<2>();

    int ndim = C.ndim();
    if (ndim != 2) {
        throw std::runtime_error("The cost matrix must be 2-dimensional");
    }
    int Nx = C.shape(0);
    int Ny = C.shape(1);

    int dim_rows = (num_valid > 0) ? num_valid : Nx;
    int dim_cols = Ny;

    auto get_cost = [&C](int x, int y) -> float { return C(x, y); };

    int *rowsol = new int[Ny];

    if (use_openmp) {
#ifdef LAP_OPENMP
        solveTableOMP<float, float>(start_time, dim_rows, dim_cols, get_cost, rowsol, use_epsilon);
#else
        throw std::runtime_error("OpenMP not enabled");
#endif
    }
    else {
        solveTable<float, float>(start_time, dim_rows, dim_cols, get_cost, rowsol, use_epsilon);
    }

    py::array_t<int32_t, py::array::c_style> result(Nx);
    auto r = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < Nx; i++) {
        r(i) = (i < dim_rows) ? rowsol[i] : -1;
    }

    delete[] rowsol;
    return result;
}

py::array_t<int32_t> solve_lap_double(
    py::array_t<double> cost_matrix,
    int num_valid = -1,
    bool use_openmp = false,
    bool use_epsilon = true
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto C = cost_matrix.mutable_unchecked<2>();

    int ndim = C.ndim();
    if (ndim != 2) {
        throw std::runtime_error("The cost matrix must be 2-dimensional");
    }
    int Nx = C.shape(0);
    int Ny = C.shape(1);

    int dim_rows = (num_valid > 0) ? num_valid : Nx;
    int dim_cols = Ny;

    auto get_cost = [&C](int x, int y) -> double { return C(x, y); };

    int *rowsol = new int[Ny];

    if (use_openmp) {
#ifdef LAP_OPENMP
        solveTableOMP<double, double>(start_time, dim_rows, dim_cols, get_cost, rowsol, use_epsilon);
#else
        throw std::runtime_error("OpenMP not enabled");
#endif
    }
    else {
        solveTable<double, double>(start_time, dim_rows, dim_cols, get_cost, rowsol, use_epsilon);
    }

    py::array_t<int32_t, py::array::c_style> result(Nx);
    auto r = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < Nx; i++) {
        r(i) = (i < dim_rows) ? rowsol[i] : -1;
    }

    delete[] rowsol;
    return result;
}

py::array_t<int32_t> solve_lap_lambda_float(
    py::array_t<float> cost_matrix,
    int num_valid = -1,
    bool use_openmp = false,
    bool use_epsilon = true
) {
    return solve_lap_float(cost_matrix, num_valid, use_openmp, use_epsilon);
}

py::array_t<int32_t> solve_lap_lambda_double(
    py::array_t<double> cost_matrix,
    int num_valid = -1,
    bool use_openmp = false,
    bool use_epsilon = true
) {
    return solve_lap_double(cost_matrix, num_valid, use_openmp, use_epsilon);
}

PYBIND11_MODULE(_lap1015, m) {
    m.doc() = "LAP1015 solver - Algorithm 1015 for Linear Assignment Problem";

    m.def("solve_lap_float", &solve_lap_float,
          py::arg("cost_matrix"),
          py::arg("num_valid") = -1,
          py::arg("use_openmp") = false,
          py::arg("use_epsilon") = true,
          "Solve LAP with single precision (float32)");

    m.def("solve_lap_double", &solve_lap_double,
          py::arg("cost_matrix"),
          py::arg("num_valid") = -1,
          py::arg("use_openmp") = false,
          py::arg("use_epsilon") = true,
          "Solve LAP with double precision (float64)");

    m.def("solve_lap_lambda_float", &solve_lap_lambda_float,
          py::arg("cost_matrix"),
          py::arg("num_valid") = -1,
          py::arg("use_openmp") = false,
          py::arg("use_epsilon") = true,
          "Solve LAP with single precision (float32) using lambda-based cost function");

    m.def("solve_lap_lambda_double", &solve_lap_lambda_double,
          py::arg("cost_matrix"),
          py::arg("num_valid") = -1,
          py::arg("use_openmp") = false,
          py::arg("use_epsilon") = true,
          "Solve LAP with double precision (float64) using lambda-based cost function");

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
