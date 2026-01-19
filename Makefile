SHELL := /bin/bash
EXECUTABLE := poetry run
CONDA_ENV ?= st-dadk

.PHONY: clean install install-dev test test-cov lint format run-local-jupyter

## Install project dependencies with Poetry
install:
	@echo "Installing project dependencies with Poetry..."
	@poetry install --with dev

## Install development dependencies (conda)
install-dev:
	@echo "Installing development dependencies in conda env: $(CONDA_ENV)..."
	@$(SHELL) envs/conda/build_conda_env.sh -c $(CONDA_ENV)

## Run tests
test:
	@echo "Running tests..."
	@$(EXECUTABLE) pytest tests/ -v

## Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	@$(EXECUTABLE) pytest tests/ -v --cov=stnf --cov-report=html --cov-report=term

## Run linters
lint:
	@echo "Running linters..."
	@$(EXECUTABLE) python -m black --check stnf scripts
	@$(EXECUTABLE) python -m isort --check-only stnf scripts
	@$(EXECUTABLE) python -m mypy stnf --ignore-missing-imports || true

## Format code
format:
	@echo "Formatting code..."
	@$(EXECUTABLE) python -m black stnf scripts
	@$(EXECUTABLE) python -m isort stnf scripts

## Start Jupyter server locally
run-local-jupyter:
	@echo "Starting local Jupyter server..."
	@$(SHELL) envs/jupyter/start_jupyter_lab.sh --port 8501

## Clean up temporary files
clean:
	@echo "Cleaning up..."
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -delete
	@rm -rf build/ dist/ .eggs/ .pytest_cache
	@find . -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@rm -f .coverage
