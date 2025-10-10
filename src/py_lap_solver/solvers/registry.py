"""Pre-instantiated solver registry for easy access to different solver configurations.

This module provides a convenient way to access different solver configurations
without having to manually instantiate them. This is especially useful for
benchmarking and comparing different solver implementations.

Example:
    >>> from py_lap_solver.solvers.registry import Solvers
    >>> result = Solvers.Lap1015OMP.solve_single(cost_matrix)
    >>> result = Solvers.Lap1015Sequential.solve_single(cost_matrix)
"""

from .batched_scipy_solver import BatchedScipySolver
from .lap1015_solver import Lap1015Solver
from .scipy_solver import ScipySolver


class SolverRegistry:
    """Registry of pre-instantiated solvers with different configurations."""

    def __init__(self):
        # Pure Python scipy solver (always available)
        self.Scipy = ScipySolver()
        self.ScipyMP8 = ScipySolver(use_python_mp=True, n_jobs=8)

        # Batched scipy solvers (with and without OpenMP)
        if BatchedScipySolver.is_available():
            self.BatchedScipyOMP = BatchedScipySolver(use_openmp=True)
            self.BatchedScipySequential = BatchedScipySolver(use_openmp=False)
        else:
            self.BatchedScipyOMP = None
            self.BatchedScipySequential = None

        # LAP1015 solvers (with and without OpenMP)
        if Lap1015Solver.is_available():
            self.Lap1015OMP = Lap1015Solver(use_openmp=True)
            self.Lap1015Sequential = Lap1015Solver(use_openmp=False)
            self.Lap1015 = self.Lap1015Sequential  # Default alias
        else:
            self.Lap1015OMP = None
            self.Lap1015Sequential = None
            self.Lap1015 = None

    def get_available_solvers(self):
        """Get a dictionary of all available solvers.

        Returns
        -------
        dict
            Dictionary mapping solver names to solver instances.
            Only includes solvers that are actually available.
        """
        solvers = {}

        # Add all solver attributes
        for name in dir(self):
            if name.startswith("_"):
                continue
            solver = getattr(self, name)
            if solver is not None and hasattr(solver, "solve_single"):
                solvers[name] = solver

        return solvers

    def get_solver_info(self):
        """Get information about all available solvers.

        Returns
        -------
        dict
            Dictionary mapping solver names to their availability and features.
        """
        info = {
            "Scipy": {
                "available": True,
                "features": ["pure_python"],
            },
            "ScipyMP8": {
                "available": True,
                "features": ["pure_python", "batch_parallel", "multiprocessing"],
            },
            "BatchedScipyOMP": {
                "available": BatchedScipySolver.is_available(),
                "has_openmp": BatchedScipySolver.has_openmp(),
                "features": (
                    ["cpp", "batch_parallel"] if BatchedScipySolver.has_openmp() else ["cpp"]
                ),
            },
            "BatchedScipySequential": {
                "available": BatchedScipySolver.is_available(),
                "features": ["cpp"],
            },
            "Lap1015OMP": {
                "available": Lap1015Solver.is_available(),
                "has_openmp": Lap1015Solver.has_openmp(),
                "has_cuda": Lap1015Solver.has_cuda(),
                "features": self._get_lap1015_features(),
            },
            "Lap1015Sequential": {
                "available": Lap1015Solver.is_available(),
                "has_cuda": Lap1015Solver.has_cuda(),
                "features": ["cpp", "optimized"] + (["cuda"] if Lap1015Solver.has_cuda() else []),
            },
        }
        return info

    def _get_lap1015_features(self):
        """Get features list for LAP1015 OpenMP solver."""
        features = ["cpp", "optimized", "intra_matrix_parallel"]
        if Lap1015Solver.has_cuda():
            features.append("cuda")
        return features

    def print_available_solvers(self):
        """Print a formatted list of all available solvers."""
        info = self.get_solver_info()

        print("Available Solvers:")
        print("-" * 60)

        for name, details in info.items():
            available = details.get("available", False)
            status = "✓" if available else "✗"
            print(f"{status} {name:25s}", end="")

            if available:
                features = details.get("features", [])
                if features:
                    print(f" [{', '.join(features)}]")
                else:
                    print()

                # Add feature flags
                flags = []
                if details.get("has_openmp"):
                    flags.append("OpenMP")
                if details.get("has_cuda"):
                    flags.append("CUDA")
                if flags:
                    print(f"  {'':25s} Features: {', '.join(flags)}")
            else:
                print(" [not available]")


# Global singleton instance
Solvers = SolverRegistry()


def get_all_available_solvers():
    """Get all available solver instances.

    Returns
    -------
    dict
        Dictionary mapping solver names to instances.
    """
    return Solvers.get_available_solvers()


def get_solver_by_name(name):
    """Get a solver by name.

    Parameters
    ----------
    name : str
        Name of the solver (e.g., 'Lap1015OMP', 'Scipy').

    Returns
    -------
    LapSolver or None
        Solver instance if available, None otherwise.
    """
    return getattr(Solvers, name, None)
