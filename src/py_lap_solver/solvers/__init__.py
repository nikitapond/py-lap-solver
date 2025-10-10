"""Solvers for Linear Assignment Problems."""

from .batched_scipy_solver import BatchedScipySolver
from .lap1015_solver import Lap1015Solver
from .registry import Solvers, get_all_available_solvers, get_solver_by_name
from .scipy_solver import ScipySolver


def get_available_solvers():
    """Get list of available solvers.

    Returns
    -------
    list of tuple
        List of (name, solver_class) tuples for all available solvers.
        ScipySolver is always available, others only if their dependencies are available.
    """
    solvers = [("ScipySolver", ScipySolver)]

    if BatchedScipySolver.is_available():
        solvers.append(("BatchedScipySolver", BatchedScipySolver))

    if Lap1015Solver.is_available():
        solvers.append(("Lap1015Solver", Lap1015Solver))

    return solvers


__all__ = [
    "ScipySolver",
    "BatchedScipySolver",
    "Lap1015Solver",
    "Solvers",
    "get_available_solvers",
    "get_all_available_solvers",
    "get_solver_by_name",
]
