"""py_lap_solver: A unified framework for Linear Assignment Problem solvers."""

from . import solvers
from .base import LapSolver


__all__ = ["LapSolver", "solvers"]
