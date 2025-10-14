import numpy as np

from ..base import LapSolver


class Lap1015Solver(LapSolver):
    """Linear Assignment Problem solver using Algorithm 1015.

    This solver uses the LAP1015 implementation, which is a highly optimized
    shortest augmenting path algorithm. It supports both single and batch solving.

    Parameters
    ----------
    maximize : bool, optional
        If True, solve the maximization problem instead of minimization.
        Default is False (minimization).
    unassigned_value : int, optional
        Value to use for unassigned rows/columns in the output arrays.
        Default is -1.
    use_openmp : bool, optional
        Whether to use OpenMP parallelization within each matrix solve.
        Default is True. If OpenMP is not available, this is ignored.
    use_epsilon : bool, optional
        Whether to use epsilon scaling (early stopping parameter).
        When True (default), the algorithm estimates an initial epsilon value
        which can improve performance for some cost structures. When False,
        skips epsilon estimation which may be faster for certain problems.
        Default is True.
    use_lambda : bool, optional
        Whether to use lambda-based cost function (SimpleCostFunction wrapper).
        May be faster due to lambda inlining. Default is False.
    """

    def __init__(
        self,
        maximize=False,
        unassigned_value=-1,
        use_openmp=True,
        use_epsilon=True,
        use_lambda=False,
        **kwargs
    ):
        super().__init__()
        self.maximize = maximize
        self.unassigned_value = unassigned_value
        self.use_openmp = use_openmp
        self.use_epsilon = use_epsilon
        self.use_lambda = use_lambda

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
        if not self._available:
            raise RuntimeError(
                "LAP1015 solver is not available. "
                "Please rebuild the package with C++ extensions enabled."
            )

        cost_matrix = np.asarray(cost_matrix)
        n_rows, n_cols = cost_matrix.shape

        # Transpose if more rows than columns (bindings expect rows <= cols)
        transposed = False
        if n_rows > n_cols:
            cost_matrix = cost_matrix.T
            transposed = True

        # Handle maximization by negating costs
        if self.maximize:
            cost_matrix = -cost_matrix.copy()

        # Pass num_valid to C++ if provided, otherwise -1 to use full matrix
        num_valid_arg = num_valid if num_valid is not None else -1

        # Determine whether to use OpenMP (only if available)
        use_openmp_arg = self.use_openmp and self._backend.HAS_OPENMP

        # Choose precision based on input dtype
        if cost_matrix.dtype == np.float32:
            if self.use_lambda:
                result = self._backend.solve_lap_lambda_float(
                    cost_matrix,
                    num_valid=num_valid_arg,
                    use_openmp=use_openmp_arg,
                    use_epsilon=self.use_epsilon,
                )
            else:
                result = self._backend.solve_lap_float(
                    cost_matrix,
                    num_valid=num_valid_arg,
                    use_openmp=use_openmp_arg,
                    use_epsilon=self.use_epsilon,
                )
        else:
            # Convert to float64 if necessary
            if cost_matrix.dtype != np.float64:
                cost_matrix = cost_matrix.astype(np.float64)
            if self.use_lambda:
                result = self._backend.solve_lap_lambda_double(
                    cost_matrix,
                    num_valid=num_valid_arg,
                    use_openmp=use_openmp_arg,
                    use_epsilon=self.use_epsilon,
                )
            else:
                result = self._backend.solve_lap_double(
                    cost_matrix,
                    num_valid=num_valid_arg,
                    use_openmp=use_openmp_arg,
                    use_epsilon=self.use_epsilon,
                )

        # If we transposed, convert col_to_row back to row_to_col
        if transposed:
            # Result is col_to_row mapping, convert to row_to_col
            col_to_row = result
            result = np.full(n_rows, -1, dtype=np.int32)
            for col_idx, row_idx in enumerate(col_to_row):
                if row_idx >= 0 and row_idx < n_rows:
                    result[row_idx] = col_idx

        # Convert unassigned values if needed
        if self.unassigned_value != -1:
            result = result.copy()
            result[result == -1] = self.unassigned_value

        return result

    def batch_solve(self, batch_cost_matrices, num_valid=None):
        """Solve multiple linear assignment problems.

        Note: Currently solves sequentially. For parallel batch solving,
        use BatchedScipySolver.

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
        if not self._available:
            raise RuntimeError(
                "LAP1015 solver is not available. "
                "Please rebuild the package with C++ extensions enabled."
            )

        batch_cost_matrices = np.asarray(batch_cost_matrices)

        if batch_cost_matrices.ndim != 3:
            raise ValueError("batch_cost_matrices must be 3D array (B, N, M)")

        batch_size, n_rows, n_cols = batch_cost_matrices.shape

        # Handle num_valid parameter
        if num_valid is None:
            num_valid_array = [None] * batch_size
        elif isinstance(num_valid, (int, np.integer)):
            num_valid_array = [num_valid] * batch_size
        else:
            num_valid_array = num_valid

        # Preallocate output array
        results = np.full((batch_size, n_rows), self.unassigned_value, dtype=np.int32)

        # Solve each problem sequentially
        for i in range(batch_size):
            results[i] = self.solve_single(batch_cost_matrices[i], num_valid_array[i])

        return results
