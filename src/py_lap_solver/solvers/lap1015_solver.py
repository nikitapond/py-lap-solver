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
    """

    def __init__(self, maximize=False, unassigned_value=-1, **kwargs):
        super().__init__()
        self.maximize = maximize
        self.unassigned_value = unassigned_value

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
        original_n_rows, original_n_cols = cost_matrix.shape
        n_rows, n_cols = original_n_rows, original_n_cols

        # Pad rectangular matrices to square to avoid issues with LAP1015 solver
        was_padded = False
        if n_rows != n_cols:
            max_dim = max(n_rows, n_cols)

            # Create padded square matrix with large cost values
            # Use a value much larger than any in the original matrix
            max_cost = np.max(np.abs(cost_matrix))
            padding_value = max_cost * 1000 + 1e10

            padded_matrix = np.full((max_dim, max_dim), padding_value, dtype=cost_matrix.dtype)
            padded_matrix[:n_rows, :n_cols] = cost_matrix

            cost_matrix = padded_matrix
            n_rows = n_cols = max_dim
            was_padded = True

        # Handle maximization by negating costs
        if self.maximize:
            cost_matrix = -cost_matrix.copy()

        # Determine num_valid for C++ call
        # Don't use num_valid for padded rectangular matrices (padding handles it)
        num_valid_arg = -1 if was_padded else (num_valid if num_valid is not None else -1)

        # Choose precision based on input dtype
        if cost_matrix.dtype == np.float32:
            result = self._backend.solve_lap_float(cost_matrix, num_valid=num_valid_arg)
        else:
            # Convert to float64 if necessary
            if cost_matrix.dtype != np.float64:
                cost_matrix = cost_matrix.astype(np.float64)
            result = self._backend.solve_lap_double(cost_matrix, num_valid=num_valid_arg)

        # Trim result back to original row dimension
        if len(result) > original_n_rows:
            result = result[:original_n_rows]

        # For rectangular matrices, filter out assignments to padded columns
        if was_padded and original_n_cols < n_cols:
            result = result.copy()
            result[result >= original_n_cols] = -1

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
