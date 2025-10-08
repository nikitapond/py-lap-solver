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
  - **ScipySolver**: Pure Python implementation using scipy's Hungarian algorithm
  - **BatchedScipySolver**: C++ implementation with OpenMP parallelization for batch processing
  - **Lap1015Solver**: Highly optimized C++ implementation (shortest augmenting path algorithm)
- **Batch Processing**: Solve multiple LAP instances efficiently with OpenMP parallelization
- **Flexible Input**: Support for square and rectangular cost matrices
- **Optional GPU Support**: CUDA support in LAP1015 (not yet fully exposed in Python bindings)

## Usage

```python
from py_lap_solver.solvers import ScipySolver, BatchedScipySolver, Lap1015Solver
import numpy as np

# Create a cost matrix
cost_matrix = np.random.rand(100, 100)

# Use scipy solver (always available)
scipy_solver = ScipySolver()
assignments = scipy_solver.solve_single(cost_matrix)

# Use batched scipy solver with OpenMP (if C++ extensions are available)
if BatchedScipySolver.is_available():
    batch_solver = BatchedScipySolver()
    batch_matrices = np.random.rand(10, 100, 100)
    batch_assignments = batch_solver.batch_solve(batch_matrices)

# Use LAP1015 solver (if C++ extensions are available)
if Lap1015Solver.is_available():
    lap_solver = Lap1015Solver()
    assignments = lap_solver.solve_single(cost_matrix)

    # Check for OpenMP support
    if Lap1015Solver.has_openmp():
        print("OpenMP parallelization available")
```

### Building with C++ Extensions

To enable the optimized C++ solvers, you need CMake and build tools:

```bash
# Install build dependencies
pip install scikit-build-core pybind11

# Build and install with C++ extensions
pip install -e . --no-build-isolation

# On macOS, you may need to install libomp for OpenMP support
brew install libomp
```

## Development

### Installation

```bash
# Install with development dependencies (includes black, ruff, pytest)
pip install -e ".[dev]"
```

### Code Formatting and Linting

The project uses `black` for code formatting and `ruff` for linting. A Makefile is provided for convenience:

```bash
# Format code with black
make format

# Lint code with ruff
make lint

# Auto-fix linting issues
make lint-fix

# Run all checks
make check

# Format, lint-fix, check, and test in one command
make all
```

Or use the tools directly:

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/
```

### Testing

```bash
# Run tests with pytest
pytest tests/

# Or use make
make test
```

## License

MIT
