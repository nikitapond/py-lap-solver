.PHONY: format lint check test clean install help

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

format:  ## Format code with black
	black src/ tests/

lint:  ## Lint code with ruff
	ruff check src/ tests/

lint-fix:  ## Lint and auto-fix issues with ruff
	ruff check --fix src/ tests/

check: lint  ## Run all checks (lint)
	@echo "All checks passed!"

test:  ## Run correctness tests
	pytest tests/

install:  ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

install-build:  ## Install package with C++ extensions
	pip install -e . --no-build-isolation

clean:  ## Clean build artifacts
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete

all: format lint-fix check test  ## Run format, lint-fix, check, and test
