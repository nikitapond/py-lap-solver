from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..base import LapSolver


def solve_single(cost_matrix, maximize, num_valid):
    """Helper function to solve a single LAP instance.

    This is defined outside the class to facilitate multiprocessing.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (N, M) where N <= M.
    maximize : bool
        If True, solve the maximization problem instead of minimization.
    num_valid : int or None
        Number of valid rows if matrix is padded. If None, uses the full matrix row size.

    Returns
    -------
    result : np.ndarray
        Array of column assignments. For rectangular matrices (N < M),
        unassigned columns are appended at the end.
    """
    cost_matrix = np.asarray(cost_matrix)
    n_cols = cost_matrix.shape[1]

    cost_matrix_to_solve = cost_matrix[:num_valid, :] if num_valid is not None else cost_matrix

    # Scipy minimizes by default, so negate for maximization
    if maximize:
        cost_matrix_to_solve = -cost_matrix_to_solve

    # Scipy returns (row_ind, col_ind) pairs
    _, col_ind = linear_sum_assignment(cost_matrix_to_solve)

    # Concatenate assigned columns + unassigned columns
    all_cols = np.arange(n_cols, dtype=np.int32)
    unassigned_cols = all_cols[~np.isin(all_cols, col_ind)]
    result = np.concatenate([col_ind, unassigned_cols])

    return result


class ScipySolver(LapSolver):
    """Linear Assignment Problem solver using scipy.optimize.linear_sum_assignment.

    Uses the Hungarian algorithm implementation from scipy.

    Parameters
    ----------
    maximize : bool, optional
        If True, solve the maximization problem instead of minimization.
        Default is False (minimization).
    use_python_mp : bool, optional
        Whether to use Python multiprocessing for batch solving. Default is False.
    n_jobs : int, optional
        Number of worker processes to use when use_python_mp is True.
        Default is 8. Ignored if use_python_mp is False.
    """

    def __init__(
        self, maximize=False, use_python_mp=False, n_jobs=8, **kwargs
    ):
        super().__init__()
        self.maximize = maximize
        self.use_python_mp = use_python_mp
        self.n_jobs = n_jobs

    def solve_single(self, cost_matrix, num_valid=None):
        """Solve a single linear assignment problem.

        Parameters
        ----------
        cost_matrix : np.ndarray
            Cost matrix of shape (N, M) where N <= M.
        num_valid : int, optional
            Number of valid rows if matrix is padded.
            If None, uses the full matrix size.

        Returns
        -------
        result : np.ndarray
            Array of column assignments.
        """
        return solve_single(
            cost_matrix,
            self.maximize,
            num_valid,
        )

    def batch_solve(self, batch_cost_matrices, num_valid=None):
        """Solve multiple linear assignment problems.

        Parameters
        ----------
        batch_cost_matrices : np.ndarray
            Batch of cost matrices of shape (B, N, M) where N <= M.
        num_valid : np.ndarray or int, optional
            Number of valid rows for each matrix.
            Can be a scalar (same for all) or array of shape (B,).

        Returns
        -------
        np.ndarray
            Array of shape (B, M) with column assignments for each problem.
        """
        batch_cost_matrices = np.asarray(batch_cost_matrices)
        batch_size, n_rows, n_cols = batch_cost_matrices.shape

        # Handle num_valid as scalar or array
        if num_valid is None:
            num_valid_array = [None] * batch_size
        elif isinstance(num_valid, (int, np.integer)):
            num_valid_array = [num_valid] * batch_size
        else:
            num_valid_array = num_valid

        if self.use_python_mp:
            # Use multiprocessing pool to solve in parallel
            args = [
                (batch_cost_matrices[i], self.maximize, num_valid_array[i])
                for i in range(batch_size)
            ]
            chunk_size = (batch_size + self.n_jobs - 1) // self.n_jobs

            with Pool(processes=self.n_jobs) as pool:
                results = pool.starmap(solve_single, args, chunksize=chunk_size)

            return np.stack(results)
        else:
            # Sequential solving
            results = np.empty((batch_size, n_cols), dtype=np.int32)

            for i in range(batch_size):
                results[i] = self.solve_single(batch_cost_matrices[i], num_valid_array[i])

            return results
