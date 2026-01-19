# Environment Setup Guide

This guide explains how to set up the development environment for ST-DADK using Conda and Poetry.

## Prerequisites

- Conda (Miniconda or Anaconda)
- Poetry (will be installed automatically if not present)

## Quick Start

### Option 1: Using Makefile (Recommended)

```bash
# Install project dependencies with Poetry
make install

# Install development dependencies in conda environment
make install-dev
```

### Option 2: Manual Setup

#### 1. Create Conda Environment

The build script will automatically create a conda environment named `st-dadk` if it doesn't exist:

```bash
bash envs/conda/build_conda_env.sh
```

Or specify a custom environment name:

```bash
bash envs/conda/build_conda_env.sh -c my-env-name
```

#### 2. Activate Environment

```bash
conda activate st-dadk
```

#### 3. Install Dependencies

```bash
poetry install --with dev
```

## Environment Structure

- `pyproject.toml`: Poetry configuration file with all dependencies
- `envs/conda/`: Conda environment management scripts
- `envs/jupyter/`: Jupyter Lab setup scripts
- `bin/`: Utility scripts (color output, exit codes)

## Available Make Commands

- `make install`: Install project dependencies with Poetry
- `make install-dev`: Install development dependencies in conda environment
- `make test`: Run tests
- `make test-cov`: Run tests with coverage report
- `make lint`: Run linters (black, isort, mypy)
- `make format`: Format code with black and isort
- `make run-local-jupyter`: Start Jupyter Lab server
- `make clean`: Clean up temporary files

## Jupyter Lab

To start Jupyter Lab with the conda environment:

```bash
make run-local-jupyter
```

Or manually:

```bash
bash envs/jupyter/start_jupyter_lab.sh
```

With custom options:

```bash
bash envs/jupyter/start_jupyter_lab.sh -k st-dadk -p 8501
```

## Troubleshooting

### Conda not found

If you get "No Conda environment found", make sure:
1. Conda is installed and in your PATH
2. Conda is initialized in your shell: `conda init bash` (or `zsh`)

### Poetry not found

Poetry will be automatically installed in the conda environment if not present.

### Permission denied

Make sure the shell scripts are executable:

```bash
chmod +x envs/**/*.sh bin/*.sh
```

## Development Workflow

1. Create/activate conda environment: `conda activate st-dadk`
2. Install dependencies: `make install`
3. Make changes to code
4. Format code: `make format`
5. Run tests: `make test`
6. Commit changes
