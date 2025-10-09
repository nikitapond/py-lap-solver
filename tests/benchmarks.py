"""Benchmarking script for LAP solvers."""

import time

from get_cost_matrices import (
    get_full_rect_matrix,
    get_full_square_matrix,
    get_masked_rect_matrix,
    get_masked_square_matrix,
)

from py_lap_solver.solvers import Solvers


def benchmark_all_solvers_single(matrix_generator, **generator_kwargs):
    """Benchmark all solvers on single problem using solve_single().

    Parameters
    ----------
    matrix_generator : callable
        Function from get_cost_matrices to generate test matrices
    **generator_kwargs : dict
        Keyword arguments to pass to matrix_generator

    Returns
    -------
    dict
        Results dictionary with timing information
    """
    # Generate the cost matrix
    result = matrix_generator(**generator_kwargs)
    if isinstance(result, tuple):
        if len(result) == 3:
            # get_masked_rect_matrix returns (matrix, n_rows, n_cols)
            cost_matrix, num_valid, _ = result
        else:
            # get_masked_square_matrix returns (matrix, num_valid)
            cost_matrix, num_valid = result
    else:
        cost_matrix = result
        num_valid = None

    results = {
        "matrix_size": cost_matrix.shape,
        "num_valid": num_valid,
        "generator": matrix_generator.__name__,
        "solvers": {},
    }

    # Benchmark each solver
    for solver_name, solver in Solvers.get_available_solvers().items():
        # Warmup
        _ = solver.solve_single(cost_matrix, num_valid=num_valid)

        # Timed run
        start_time = time.perf_counter()
        assignment = solver.solve_single(cost_matrix, num_valid=num_valid)
        end_time = time.perf_counter()

        elapsed = end_time - start_time

        results["solvers"][solver_name] = {"time": elapsed, "assignment": assignment}

    return results


def benchmark_all_solvers_batch_single(matrix_generator, **generator_kwargs):
    """Benchmark all solvers on batch of size 1 using batch_solve().

    Parameters
    ----------
    matrix_generator : callable
        Function from get_cost_matrices to generate test matrices
    **generator_kwargs : dict
        Keyword arguments to pass to matrix_generator

    Returns
    -------
    dict
        Results dictionary with timing information
    """
    # Generate the cost matrix as a batch of size 1
    result = matrix_generator(batch_size=1, **generator_kwargs)
    if isinstance(result, tuple):
        if len(result) == 3:
            # get_masked_rect_matrix returns (matrix, n_rows, n_cols)
            batch_matrices, num_valid, _ = result
        else:
            # get_masked_square_matrix returns (matrix, num_valid)
            batch_matrices, num_valid = result
    else:
        batch_matrices = result
        num_valid = None

    results = {
        "matrix_size": batch_matrices.shape[1:],
        "batch_size": 1,
        "num_valid": num_valid,
        "generator": matrix_generator.__name__,
        "solvers": {},
    }

    # Benchmark each solver
    for solver_name, solver in Solvers.get_available_solvers().items():
        # Warmup
        _ = solver.batch_solve(batch_matrices, num_valid=num_valid)

        # Timed run
        start_time = time.perf_counter()
        assignments = solver.batch_solve(batch_matrices, num_valid=num_valid)
        end_time = time.perf_counter()

        elapsed = end_time - start_time

        results["solvers"][solver_name] = {"time": elapsed, "assignments": assignments}

    return results


def benchmark_all_solvers_batch(matrix_generator, batch_size, **generator_kwargs):
    """Benchmark all solvers on a batch of problems.

    Parameters
    ----------
    matrix_generator : callable
        Function from get_cost_matrices to generate test matrices
    batch_size : int
        Number of problems in the batch
    **generator_kwargs : dict
        Keyword arguments to pass to matrix_generator

    Returns
    -------
    dict
        Results dictionary with timing information
    """
    # Generate the batch of cost matrices
    result = matrix_generator(batch_size=batch_size, **generator_kwargs)
    if isinstance(result, tuple):
        if len(result) == 3:
            # get_masked_rect_matrix returns (matrix, n_rows, n_cols)
            batch_matrices, num_valid, _ = result
        else:
            # get_masked_square_matrix returns (matrix, num_valid)
            batch_matrices, num_valid = result
    else:
        batch_matrices = result
        num_valid = None

    results = {
        "matrix_size": batch_matrices.shape[1:],
        "batch_size": batch_size,
        "num_valid": num_valid,
        "generator": matrix_generator.__name__,
        "solvers": {},
    }

    # Benchmark each solver
    for solver_name, solver in Solvers.get_available_solvers().items():
        # Warmup
        _ = solver.batch_solve(batch_matrices, num_valid=num_valid)

        # Timed run
        start_time = time.perf_counter()
        assignments = solver.batch_solve(batch_matrices, num_valid=num_valid)
        end_time = time.perf_counter()

        elapsed = end_time - start_time

        results["solvers"][solver_name] = {
            "time": elapsed,
            "throughput": batch_size / elapsed,  # problems per second
            "assignments": assignments,
        }

    return results


def print_single_results(results):
    """Pretty print results from single problem benchmarks."""
    print(f"\n{'='*70}")
    print(f"Generator: {results['generator']}")
    print(f"Matrix size: {results['matrix_size']}")
    if results["num_valid"] is not None:
        print(f"Valid size: {results['num_valid']}")
    print(f"{'='*70}")

    for solver_name, solver_results in results["solvers"].items():
        print(f"{solver_name:25s}: {solver_results['time']*1000:8.3f} ms")


def print_batch_results(results):
    """Pretty print results from batch benchmarks."""
    print(f"\n{'='*70}")
    print(f"Generator: {results['generator']}")
    print(f"Matrix size: {results['matrix_size']}, Batch size: {results['batch_size']}")
    if results["num_valid"] is not None:
        print(f"Valid size: {results['num_valid']}")
    print(f"{'='*70}")

    for solver_name, solver_results in results["solvers"].items():
        time_ms = solver_results["time"] * 1000
        if "throughput" in solver_results:
            throughput = solver_results["throughput"]
            print(f"{solver_name:25s}: {time_ms:8.3f} ms  ({throughput:8.1f} problems/sec)")
        else:
            print(f"{solver_name:25s}: {time_ms:8.3f} ms")


def run_full_benchmark_suite():
    """Run comprehensive benchmark suite."""
    print("\n" + "=" * 70)
    print("LAP Solver Benchmark Suite")
    print("=" * 70)

    # Matrix sizes to test
    square_sizes = [10, 50, 100, 500, 1000]
    rect_sizes = [(10, 15), (50, 100), (100, 50), (100, 200)]
    batch_sizes = [1, 5, 10, 50, 100, 1000]

    # Large matrix sizes for small batch tests
    large_sizes = [1000, 5000, 10000]

    # ========================================================================
    # Single problem benchmarks (using solve_single)
    # ========================================================================
    print("\n" + "=" * 70)
    print("SINGLE PROBLEM BENCHMARKS (solve_single)")
    print("=" * 70)

    # Full square matrices
    for size in square_sizes:
        results = benchmark_all_solvers_single(get_full_square_matrix, size=size)
        print_single_results(results)

    # Masked square matrices
    for size in [50, 100, 500]:
        results = benchmark_all_solvers_single(get_masked_square_matrix, size=size)
        print_single_results(results)

    # Full rectangular matrices
    for n_rows, n_cols in rect_sizes:
        results = benchmark_all_solvers_single(get_full_rect_matrix, n_rows=n_rows, n_cols=n_cols)
        print_single_results(results)

    # Masked rectangular matrices
    for n_rows, n_cols in [(50, 100), (100, 200)]:
        results = benchmark_all_solvers_single(get_masked_rect_matrix, n_rows=n_rows, n_cols=n_cols)
        print_single_results(results)

    # ========================================================================
    # Batch size = 1 benchmarks (using batch_solve)
    # ========================================================================
    print("\n" + "=" * 70)
    print("BATCH SIZE = 1 BENCHMARKS (batch_solve)")
    print("=" * 70)

    for size in [50, 100, 500]:
        results = benchmark_all_solvers_batch_single(get_full_square_matrix, size=size)
        print_batch_results(results)

    # ========================================================================
    # Variable batch size benchmarks
    # ========================================================================
    print("\n" + "=" * 70)
    print("VARIABLE BATCH SIZE BENCHMARKS")
    print("=" * 70)

    # Square matrices with varying batch sizes
    for size in [50, 100, 500]:
        print(f"\n--- Square matrices ({size}x{size}) ---")
        for batch_size in batch_sizes:
            if batch_size == 1:
                continue  # Already tested above
            results = benchmark_all_solvers_batch(
                get_full_square_matrix, batch_size=batch_size, size=size
            )
            print_batch_results(results)

    # Rectangular matrices with varying batch sizes
    for n_rows, n_cols in [(50, 100), (100, 50)]:
        print(f"\n--- Rectangular matrices ({n_rows}x{n_cols}) ---")
        for batch_size in [10, 50, 100]:
            results = benchmark_all_solvers_batch(
                get_full_rect_matrix, batch_size=batch_size, n_rows=n_rows, n_cols=n_cols
            )
            print_batch_results(results)

    # Masked matrices with varying batch sizes
    print("\n--- Masked square matrices (100x100) ---")
    for batch_size in [10, 50, 100]:
        results = benchmark_all_solvers_batch(
            get_masked_square_matrix, batch_size=batch_size, size=100
        )
        print_batch_results(results)

    # ========================================================================
    # Large matrices with small batches
    # ========================================================================
    print("\n" + "=" * 70)
    print("LARGE MATRICES WITH SMALL BATCHES")
    print("=" * 70)

    for size in large_sizes:
        print(f"\n--- Large square matrices ({size}x{size}) with small batches ---")

        results = benchmark_all_solvers_single(get_full_square_matrix, size=size)
        print_single_results(results)

    print("\n" + "=" * 70)
    print("Benchmark suite complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Print available solvers
    print("\n" + "=" * 70)
    print("LAP Solver Benchmark - Available Solvers")
    print("=" * 70)
    Solvers.print_available_solvers()

    # Run the full benchmark suite
    run_full_benchmark_suite()
