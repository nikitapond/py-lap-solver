"""Base class for Linear Assignment Problem solvers."""


class LapSolver:
    """Abstract base class for Linear Assignment Problem (LAP) solvers.

    All solver implementations should inherit from this class and implement
    the solve_single and batch_solve methods.

    The LAP seeks to find an optimal assignment between two sets given a cost matrix,
    minimizing (or maximizing) the total cost of the assignment.
    """

    def __init__(self):
        """Initialize the LAP solver.

        Subclasses should override this to accept solver-specific configuration
        via kwargs.
        """
        pass

    def solve_single(self, cost_matrix, num_valid=None):
        """Solve a single linear assignment problem.

        Args:
            cost_matrix (array-like): Cost matrix of shape (N, M) where element [i, j]
                                     represents the cost of assigning row i to column j.
            num_valid (int, optional): Number of valid rows/columns if the cost matrix
                                      is padded. If None, the full matrix is used.
                                      This allows handling variable-sized problems with
                                      a fixed-size padded matrix.

        Returns:
            np.ndarray: Array of shape (N,) where element i contains the column index
                       assigned to row i. Unassigned rows have value -1.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement solve_single")

    def batch_solve(self, batch_cost_matrices, num_valid=None):
        """Solve multiple linear assignment problems.

        Args:
            batch_cost_matrices (array-like): Batch of cost matrices of shape (B, N, M)
                                             where B is the batch size.
            num_valid (int or array-like, optional): Number of valid rows/columns for each
                                                    matrix. Can be:
                                                    - None: use full matrix size for all
                                                    - int: same value for all matrices
                                                    - array of shape (B,): different value per matrix

        Returns:
            np.ndarray: Array of shape (B, N) where element [b, i] contains the column
                       index assigned to row i in batch element b. Unassigned rows have
                       value -1.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement batch_solve")
