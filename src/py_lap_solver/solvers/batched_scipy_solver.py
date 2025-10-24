import numpy as np

from ..base import LapSolver


class BatchedScipySolver(LapSolver):
    """Linear Assignment Problem solver using batched scipy with OpenMP parallelization.

    This solver uses the scipy Hungarian algorithm implementation with OpenMP
    parallelization across the batch dimension for improved performance on
    multi-core systems.

    Parameters
    ----------
    maximize : bool, optional
        If True, solve the maximization problem instead of minimization.
        Default is False (minimization).
    unassigned_value : int, optional
        Value to use for unassigned rows/columns in the output arrays.
        Default is -1.
    use_openmp : bool, optional
        Whether to use OpenMP parallelization. Default is True.
        If OpenMP is not available, this is ignored.
    """

    def __init__(self, maximize=False, use_openmp=True, **kwargs):
        super().__init__()
        self.maximize = maximize
        self.use_openmp = use_openmp

        # Try to import the C++ extension
        try:
            from py_lap_solver import _batched_scipy_lap

            self._backend = _batched_scipy_lap
            self._available = True
        except ImportError:
            self._backend = None
            self._available = False

    @staticmethod
    def is_available():
        """Check if the batched scipy solver is available."""
        try:
            from py_lap_solver import _batched_scipy_lap  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def has_openmp():
        """Check if OpenMP support is available."""
        try:
            from py_lap_solver import _batched_scipy_lap

            return _batched_scipy_lap.HAS_OPENMP
        except ImportError:
            return False

    def solve_single(self, cost_matrix, num_valid=None):
        """Solve a single linear assignment problem.

        Note: This solver is optimized for batch processing. For single problems,
        consider using ScipySolver instead.

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
            Array of shape (M,) with column assignments.
        """
        if not self._available:
            raise RuntimeError(
                "Batched scipy solver is not available. "
                "Please rebuild the package with C++ extensions enabled."
            )

        # Convert single problem to batch of size 1
        cost_matrix = np.asarray(cost_matrix)
        n_cols = cost_matrix.shape[1]
        batch_cost = cost_matrix[np.newaxis, :, :]

        num_valid_arg = None if num_valid is None else num_valid

        # Choose precision based on input dtype
        if cost_matrix.dtype == np.float32:
            col_ind = self._backend.solve_batched_lap_float(
                batch_cost,
                maximize=self.maximize,
                num_valid=num_valid_arg,
                unassigned_value=-1,
                use_openmp=self.use_openmp,
            )[0]
        else:
            col_ind = self._backend.solve_batched_lap_double(
                batch_cost,
                maximize=self.maximize,
                num_valid=num_valid_arg,
                unassigned_value=-1,
                use_openmp=self.use_openmp,
            )[0]

        # Append unassigned columns
        all_cols = np.arange(n_cols, dtype=np.int32)
        unassigned_cols = all_cols[~np.isin(all_cols, col_ind[col_ind >= 0])]
        result = np.concatenate([col_ind[col_ind >= 0], unassigned_cols])

        return result

    def batch_solve(self, batch_cost_matrices, num_valid=None):
        """Solve multiple linear assignment problems with OpenMP parallelization.

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
        if not self._available:
            raise RuntimeError(
                "Batched scipy solver is not available. "
                "Please rebuild the package with C++ extensions enabled."
            )

        batch_cost_matrices = np.asarray(batch_cost_matrices)

        if batch_cost_matrices.ndim != 3:
            raise ValueError("batch_cost_matrices must be 3D array (B, N, M)")

        batch_size, n_rows, n_cols = batch_cost_matrices.shape

        # Handle num_valid parameter
        num_valid_arg = None
        if num_valid is not None:
            if isinstance(num_valid, (int, np.integer)):
                num_valid_arg = int(num_valid)
            else:
                num_valid_arg = np.asarray(num_valid, dtype=np.int64)

        # Choose precision based on input dtype
        if batch_cost_matrices.dtype == np.float32:
            batch_col_ind = self._backend.solve_batched_lap_float(
                batch_cost_matrices,
                maximize=self.maximize,
                num_valid=num_valid_arg,
                unassigned_value=-1,
                use_openmp=self.use_openmp,
            )
        else:
            batch_col_ind = self._backend.solve_batched_lap_double(
                batch_cost_matrices,
                maximize=self.maximize,
                num_valid=num_valid_arg,
                unassigned_value=-1,
                use_openmp=self.use_openmp,
            )

        # Append unassigned columns for each problem
        results = np.empty((batch_size, n_cols), dtype=np.int32)
        all_cols = np.arange(n_cols, dtype=np.int32)

        for i in range(batch_size):
            col_ind = batch_col_ind[i]
            valid_assignments = col_ind[col_ind >= 0]
            unassigned_cols = all_cols[~np.isin(all_cols, valid_assignments)]
            results[i] = np.concatenate([valid_assignments, unassigned_cols])

        return results
