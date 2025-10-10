"""Test that all solvers produce results consistent with scipy."""

import numpy as np
import pytest
from get_cost_matrices import (
    get_full_rect_matrix,
    get_full_square_matrix,
    get_masked_square_matrix,
    get_padded_square_to_rect_matrix,
)
from scipy.optimize import linear_sum_assignment

from py_lap_solver.solvers import Solvers


def get_all_solver_instances():
    """Get all available solver instances from the registry.

    Returns
    -------
    list of tuple
        List of (name, solver_instance) tuples for all available solvers.
    """
    return list(Solvers.get_available_solvers().items())


def scipy_reference(cost_matrix):
    """Reference implementation using scipy directly.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (N, M).

    Returns
    -------
    row_to_col : np.ndarray
        Array of shape (N,) where row_to_col[i] gives the column assigned to row i.
        Unassigned rows have value -1.
    """
    n_rows = cost_matrix.shape[0]

    # Get scipy's assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Convert to row_to_col format
    row_to_col = np.full(n_rows, -1, dtype=np.int32)
    row_to_col[row_ind] = col_ind

    return row_to_col


def compute_assignment_cost(cost_matrix, row_to_col):
    """Compute total cost of an assignment.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (N, M).
    row_to_col : np.ndarray
        Array of shape (N,) where row_to_col[i] gives the column assigned to row i.

    Returns
    -------
    float
        Total cost of the assignment.
    """
    total_cost = 0.0
    for row, col in enumerate(row_to_col):
        if col >= 0:  # Skip unassigned rows
            total_cost += cost_matrix[row, col]
    return total_cost


# Fixtures for different matrix types
@pytest.fixture(params=[10, 50, 100, 500])
def full_square_problem(request):
    """Fixture providing (cost_matrix, scipy_solution) for full square matrices."""
    size = request.param
    cost_matrix = get_full_square_matrix(size)
    ref_row_to_col = scipy_reference(cost_matrix)
    ref_cost = compute_assignment_cost(cost_matrix, ref_row_to_col)

    return {
        "cost_matrix": cost_matrix,
        "ref_row_to_col": ref_row_to_col,
        "ref_cost": ref_cost,
        "num_valid": None,
    }


@pytest.fixture(params=[20, 50])
def masked_square_problem(request):
    """Fixture providing (cost_matrix, scipy_solution) for masked square matrices."""
    size = request.param
    cost_matrix, num_valid = get_masked_square_matrix(size)

    # Reference uses only the valid portion
    valid_matrix = cost_matrix[:num_valid, :num_valid]
    ref_row_to_col = scipy_reference(valid_matrix)
    ref_cost = compute_assignment_cost(valid_matrix, ref_row_to_col)

    return {
        "cost_matrix": cost_matrix,
        "ref_row_to_col": ref_row_to_col,
        "ref_cost": ref_cost,
        "num_valid": num_valid,
    }


@pytest.fixture(params=[(10, 15), (20, 10), (100, 50)])
def full_rect_problem(request):
    """Fixture providing (cost_matrix, scipy_solution) for full rectangular matrices."""
    n_rows, n_cols = request.param
    cost_matrix = get_full_rect_matrix(n_rows, n_cols)
    ref_row_to_col = scipy_reference(cost_matrix)
    ref_cost = compute_assignment_cost(cost_matrix, ref_row_to_col)

    return {
        "cost_matrix": cost_matrix,
        "ref_row_to_col": ref_row_to_col,
        "ref_cost": ref_cost,
        "num_valid": None,
    }


@pytest.fixture(params=[(5, 50), (10, 100)])
def batch_square_problem(request):
    """Fixture providing batch of square matrices with scipy solutions."""
    batch_size, matrix_size = request.param
    batch_matrices = get_full_square_matrix(matrix_size, batch_size=batch_size)

    # Compute reference for each matrix
    ref_solutions = []
    for i in range(batch_size):
        ref_row_to_col = scipy_reference(batch_matrices[i])
        ref_cost = compute_assignment_cost(batch_matrices[i], ref_row_to_col)
        ref_solutions.append(
            {
                "ref_row_to_col": ref_row_to_col,
                "ref_cost": ref_cost,
            }
        )

    return {
        "batch_matrices": batch_matrices,
        "ref_solutions": ref_solutions,
    }


@pytest.fixture(params=[(10, 5), (20, 10), (50, 30)])
def padded_square_to_rect_problem(request):
    """Fixture for padded square matrices with num_valid specifying valid rows.

    Tests the case where we have N ground truth objects (rows) but the matrix is
    padded to (M, M) where M > N. We use num_valid=N to indicate only the first
    N rows are valid, solving the (N, M) assignment problem.
    """
    full_size, num_valid_rows = request.param
    cost_matrix, num_valid = get_padded_square_to_rect_matrix(full_size, num_valid_rows)

    # Reference uses only the valid rows (num_valid rows, all columns)
    valid_matrix = cost_matrix[:num_valid, :]
    ref_row_to_col = scipy_reference(valid_matrix)
    ref_cost = compute_assignment_cost(valid_matrix, ref_row_to_col)

    return {
        "cost_matrix": cost_matrix,
        "ref_row_to_col": ref_row_to_col,
        "ref_cost": ref_cost,
        "num_valid": num_valid,
        "full_size": full_size,
    }


@pytest.mark.parametrize("solver_name,solver_instance", get_all_solver_instances())
class TestSolverConsistency:
    """Test that all solvers match scipy reference."""

    def test_full_square_matrices(self, solver_name, solver_instance, full_square_problem):
        """Test full square matrices of various sizes."""
        problem = full_square_problem
        solver = solver_instance

        row_to_col = solver.solve_single(problem["cost_matrix"])
        cost = compute_assignment_cost(problem["cost_matrix"], row_to_col)

        # Check costs match
        assert np.isclose(
            cost, problem["ref_cost"], atol=1e-6
        ), f"{solver_name}: Cost mismatch: {cost} vs {problem['ref_cost']}"

        # Check shapes
        n_rows = problem["cost_matrix"].shape[0]
        assert row_to_col.shape == (n_rows,)

    def test_masked_square_matrices(self, solver_name, solver_instance, masked_square_problem):
        """Test masked square matrices (padded with invalid entries)."""
        problem = masked_square_problem
        solver = solver_instance

        row_to_col = solver.solve_single(problem["cost_matrix"], num_valid=problem["num_valid"])

        # Only check the valid portion
        num_valid = problem["num_valid"]
        valid_matrix = problem["cost_matrix"][:num_valid, :num_valid]
        cost = compute_assignment_cost(valid_matrix, row_to_col[:num_valid])

        # Check costs match
        assert np.isclose(
            cost, problem["ref_cost"], atol=1e-6
        ), f"{solver_name}: Cost mismatch: {cost} vs {problem['ref_cost']}"

        # Check shapes match original matrix
        n_rows = problem["cost_matrix"].shape[0]
        assert row_to_col.shape == (n_rows,)

    def test_full_rect_matrices(self, solver_name, solver_instance, full_rect_problem):
        """Test full rectangular matrices."""
        problem = full_rect_problem
        solver = solver_instance

        row_to_col = solver.solve_single(problem["cost_matrix"])
        cost = compute_assignment_cost(problem["cost_matrix"], row_to_col)

        # Check costs match
        assert np.isclose(
            cost, problem["ref_cost"], atol=1e-6
        ), f"{solver_name}: Cost mismatch: {cost} vs {problem['ref_cost']}"

        # Check shapes
        n_rows = problem["cost_matrix"].shape[0]
        assert row_to_col.shape == (n_rows,)

    def test_batch_square(self, solver_name, solver_instance, batch_square_problem):
        """Test batch solving with square matrices."""
        problem = batch_square_problem
        solver = solver_instance

        results = solver.batch_solve(problem["batch_matrices"])

        assert results.shape[0] == len(problem["ref_solutions"])

        for i in range(len(problem["ref_solutions"])):
            ref = problem["ref_solutions"][i]
            cost = compute_assignment_cost(problem["batch_matrices"][i], results[i])

            assert np.isclose(
                cost, ref["ref_cost"], atol=1e-6
            ), f"{solver_name}: Batch {i} cost mismatch"

    def test_padded_square_to_rect(
        self, solver_name, solver_instance, padded_square_to_rect_problem
    ):
        """Test padded square matrices where num_valid specifies valid rows only.

        This tests the case where we have N ground truth objects (rows) but the matrix
        is padded to (M, M) where M > N. num_valid=N means we only use the first N rows,
        resulting in a (N, M) assignment problem.
        """
        problem = padded_square_to_rect_problem
        solver = solver_instance

        row_to_col = solver.solve_single(problem["cost_matrix"], num_valid=problem["num_valid"])

        # Compute cost on the valid rectangular portion (num_valid rows, all columns)
        valid_matrix = problem["cost_matrix"][: problem["num_valid"], :]
        cost = compute_assignment_cost(valid_matrix, row_to_col[: problem["num_valid"]])

        # Check costs match
        assert np.isclose(
            cost, problem["ref_cost"], atol=1e-6
        ), f"{solver_name}: Cost mismatch: {cost} vs {problem['ref_cost']}"

        # Check shapes match original matrix
        assert row_to_col.shape == (problem["full_size"],)
