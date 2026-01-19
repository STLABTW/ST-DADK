SHELL := /bin/bash
EXECUTABLE := poetry run
CONDA_ENV ?= st-dadk

.PHONY: help clean install install-dev test test-cov lint format run-local-jupyter train grid-search analyze resume

.DEFAULT_GOAL := help

## Show this help message
help:
	@echo "ST-DADK Makefile Commands"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make install          Install project dependencies with Poetry"
	@echo "  make install-dev      Install development dependencies in conda environment"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linters (black, isort, mypy)"
	@echo "  make format           Format code with black and isort"
	@echo "  make test             Run tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo ""
	@echo "Training & Experiments:"
	@echo "  make train            Train model with default config"
	@echo "  make grid-search      Run grid search experiments"
	@echo "  make analyze          Analyze grid search results"
	@echo ""
	@echo "Utilities:"
	@echo "  make run-local-jupyter Start Jupyter Lab server"
	@echo "  make clean            Clean up temporary files"
	@echo ""
	@echo "For detailed script usage, see scripts/README.md"

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

## Train model with default config
train:
	@echo "Training model..."
	@$(EXECUTABLE) python scripts/train_st_interp.py --config configs/config_st_interp.yaml

## Run grid search experiments
grid-search:
	@echo "Running grid search..."
	@$(EXECUTABLE) python scripts/run_grid_search.py --config configs/config_st_interp.yaml --parallel --n_jobs 10

## Analyze grid search results (requires results directory)
analyze:
	@echo "Analyzing grid search results..."
	@echo "Usage: make analyze RESULTS_DIR=results/YYYYMMDD_grid_search"
	@if [ -z "$(RESULTS_DIR)" ]; then \
		echo "[ERROR] Please specify RESULTS_DIR"; \
		exit 1; \
	fi
	@$(EXECUTABLE) python scripts/analyze_grid_search.py $(RESULTS_DIR)

## Clean up temporary files
clean:
	@echo "Cleaning up..."
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -delete
	@rm -rf build/ dist/ .eggs/ .pytest_cache
	@find . -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@rm -f .coverage
