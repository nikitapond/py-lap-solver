# py-lap-solver

A unified Python framework for Linear Assignment Problem (LAP) solvers.

## Overview

`py-lap-solver` provides a common interface for multiple LAP solver implementations, ranging from pure Python (scipy) to optimized C++ implementations with OpenMP and CUDA support.

The Linear Assignment Problem seeks to find an optimal assignment between two sets given a cost matrix, minimizing (or maximizing) the total cost of the assignment.

## Installation

```bash
# Install the package in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Features

- **Unified Interface**: Common API across all solver implementations
- **Multiple Backends**:
  - ScipySolver: Pure Python implementation using scipy
  - SgutheSolver: Optimized C++ implementation with optional OpenMP parallelization
- **Batch Processing**: Solve multiple LAP instances efficiently
- **Optional GPU Support**: CUDA acceleration (when available)

## Usage

```python
from py_lap_solver.solvers import ScipySolver, SgutheSolver
import numpy as np

# Create a cost matrix
cost_matrix = np.random.rand(100, 100)

# Use scipy solver
scipy_solver = ScipySolver()
row_ind, col_ind = scipy_solver.solve_single(cost_matrix)

# Use C++ solver (if available)
if SgutheSolver.is_available():
    sguthe_solver = SgutheSolver(use_openmp=True)
    row_ind, col_ind = sguthe_solver.solve_single(cost_matrix)
```

## Testing

```bash
python benchmarks/test_correctness.py
```

## License

MIT
