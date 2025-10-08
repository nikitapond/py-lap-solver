import numpy as np
from scipy.optimize import linear_sum_assignment
from ..base import LapSolver


class ScipySolver(LapSolver):
    """Linear Assignment Problem solver using scipy.optimize.linear_sum_assignment.

    This solver uses the Hungarian algorithm implementation from scipy.
    It's reliable and well-tested, suitable for small to medium-sized problems.

    Parameters
    ----------
    maximize : bool, optional
        If True, solve the maximization problem instead of minimization.
        Default is False (minimization).
    unassigned_value : int, optional
        Value to use for unassigned rows/columns in the output arrays.
        Default is -1.
    """

    def __init__(self, maximize=False, unassigned_value=-1, **kwargs):
        super().__init__()
        self.maximize = maximize
        self.unassigned_value = unassigned_value

    def solve_single(self, cost_matrix, num_valid=None):
        """Solve a single linear assignment problem.

        Parameters
        ----------
        cost_matrix : np.ndarray
            Cost matrix of shape (N, M).
        num_valid : int, optional
            Number of valid rows/cols if matrix is padded.
            If None, uses the full matrix size.

        Returns
        -------
        row_to_col : np.ndarray
            Array of shape (N,) where row_to_col[i] gives the column assigned to row i.
            Unassigned rows have value `unassigned_value`.
        """
        cost_matrix = np.asarray(cost_matrix)
        n_rows, n_cols = cost_matrix.shape

        # Handle num_valid by slicing the matrix
        if num_valid is not None:
            cost_matrix_to_solve = cost_matrix[:num_valid, :num_valid]
        else:
            cost_matrix_to_solve = cost_matrix

        # Scipy minimizes by default, so negate for maximization
        if self.maximize:
            cost_matrix_to_solve = -cost_matrix_to_solve

        # Scipy returns (row_ind, col_ind) pairs
        row_ind, col_ind = linear_sum_assignment(cost_matrix_to_solve)

        # Convert to full-size array matching input row dimension
        row_to_col = np.full(n_rows, self.unassigned_value, dtype=np.int32)

        # Fill in the assignments
        row_to_col[row_ind] = col_ind

        return row_to_col

    def batch_solve(self, batch_cost_matrices, num_valid=None):
        """Solve multiple linear assignment problems.

        Parameters
        ----------
        batch_cost_matrices : np.ndarray
            Batch of cost matrices of shape (B, N, M).
        num_valid : np.ndarray or int, optional
            Number of valid rows/cols for each matrix.
            Can be a scalar (same for all) or array of shape (B,).

        Returns
        -------
        np.ndarray
            Array of shape (B, N) where element [b, i] gives the column assigned
            to row i in batch element b. Unassigned rows have value `unassigned_value`.
        """
        batch_cost_matrices = np.asarray(batch_cost_matrices)
        batch_size, n_rows, _ = batch_cost_matrices.shape

        # Handle num_valid as scalar or array
        if num_valid is None:
            num_valid_array = [None] * batch_size
        elif isinstance(num_valid, (int, np.integer)):
            num_valid_array = [num_valid] * batch_size
        else:
            num_valid_array = num_valid

        # Preallocate output array
        results = np.full((batch_size, n_rows), self.unassigned_value, dtype=np.int32)

        for i in range(batch_size):
            results[i] = self.solve_single(batch_cost_matrices[i], num_valid_array[i])

        return results
