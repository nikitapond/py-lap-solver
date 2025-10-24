import numpy as np

from ..base import LapSolver


class Lap1015Solver(LapSolver):
    """Linear Assignment Problem solver using Algorithm 1015.

    Minimal wrapper matching hepattn's implementation for maximum performance.
    Expects cost matrices where rows <= cols.

    Parameters
    ----------
    use_openmp : bool, optional
        Whether to use OpenMP parallelization. Default is False.
    use_epsilon : bool, optional
        Whether to use epsilon scaling. Default is True.
    """

    def __init__(
        self,
        use_openmp=False,
        use_epsilon=True,
        **kwargs,
    ):
        super().__init__()
        self.use_openmp = use_openmp
        self.use_epsilon = use_epsilon

        # Try to import the C++ extension
        try:
            from py_lap_solver import _lap1015

            self._backend = _lap1015
            self._available = True
        except ImportError:
            self._backend = None
            self._available = False

    @staticmethod
    def is_available():
        """Check if the LAP1015 solver is available."""
        try:
            from py_lap_solver import _lap1015  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def has_openmp():
        """Check if OpenMP support is available."""
        try:
            from py_lap_solver import _lap1015

            return _lap1015.HAS_OPENMP
        except ImportError:
            return False

    @staticmethod
    def has_cuda():
        """Check if CUDA support is available."""
        try:
            from py_lap_solver import _lap1015

            return _lap1015.HAS_CUDA
        except ImportError:
            return False

    def solve_single(self, cost_matrix, num_valid=None):
        """Solve a single linear assignment problem.

        Parameters
        ----------
        cost_matrix : np.ndarray
            Cost matrix of shape (N, M) where N <= M.
        num_valid : int, optional
            Number of valid rows if matrix is padded.

        Returns
        -------
        result : np.ndarray
            Array of shape (M,) with column assignments for each row,
            followed by unassigned columns.
        """
        if not self._available:
            raise RuntimeError(
                "LAP1015 solver is not available. "
                "Please rebuild the package with C++ extensions enabled."
            )

        cost_matrix = np.asarray(cost_matrix)
        n_rows, n_cols = cost_matrix.shape

        if n_rows > n_cols:
            raise ValueError(
                f"Cost matrix must have rows <= cols, got {n_rows}x{n_cols}. "
                "Transpose your matrix before calling this solver."
            )

        # Convert to float32 if needed (lap1015 uses float32)
        if cost_matrix.dtype != np.float32:
            cost_matrix = cost_matrix.astype(np.float32)

        num_valid_arg = num_valid if num_valid is not None else -1
        use_openmp_arg = self.use_openmp and self._backend.HAS_OPENMP

        # Call C++ backend - returns assignments for first N rows
        col_ind = self._backend.solve_lap_float(
            cost_matrix,
            num_valid=num_valid_arg,
            use_openmp=use_openmp_arg,
            use_epsilon=self.use_epsilon,
        )

        # Append unassigned columns to match Scipy's format
        all_cols = np.arange(n_cols, dtype=np.int32)
        unassigned_cols = all_cols[~np.isin(all_cols, col_ind)]
        result = np.concatenate([col_ind, unassigned_cols])

        return result

    def batch_solve(self, batch_cost_matrices, num_valid=None):
        """Solve multiple linear assignment problems sequentially.

        Parameters
        ----------
        batch_cost_matrices : np.ndarray
            Batch of cost matrices of shape (B, N, M) where N <= M.
        num_valid : np.ndarray or int, optional
            Number of valid rows for each matrix.

        Returns
        -------
        np.ndarray
            Array of shape (B, M) with column assignments for each problem.
        """
        if not self._available:
            raise RuntimeError(
                "LAP1015 solver is not available. "
                "Please rebuild the package with C++ extensions enabled."
            )

        batch_cost_matrices = np.asarray(batch_cost_matrices)

        if batch_cost_matrices.ndim != 3:
            raise ValueError("batch_cost_matrices must be 3D array (B, N, M)")

        batch_size, n_rows, n_cols = batch_cost_matrices.shape

        if n_rows > n_cols:
            raise ValueError(
                f"Cost matrices must have rows <= cols, got {n_rows}x{n_cols}. "
                "Transpose your matrices before calling this solver."
            )

        # Handle num_valid parameter
        if num_valid is None:
            num_valid_array = [None] * batch_size
        elif isinstance(num_valid, (int, np.integer)):
            num_valid_array = [num_valid] * batch_size
        else:
            num_valid_array = num_valid

        # Preallocate output array
        results = np.empty((batch_size, n_cols), dtype=np.int32)

        # Solve each problem sequentially
        for i in range(batch_size):
            results[i] = self.solve_single(batch_cost_matrices[i], num_valid_array[i])

        return results
