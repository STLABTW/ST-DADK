# ST-DADK

Spatio-Temporal Interpolation Model for KAUST Competition

## Quick Start

### Prerequisites

- Conda (Miniconda or Anaconda)
- Poetry (will be installed automatically if not present)

### Installation

#### Option 1: Using Makefile (Recommended)

```bash
# Install project dependencies with Poetry
make install

# Install development dependencies in conda environment
make install-dev
```

#### Option 2: Manual Setup

```bash
# Create and build conda environment
bash envs/conda/build_conda_env.sh

# Activate environment
conda activate st-dadk

# Install dependencies
poetry install --with dev
```

## Available Make Commands

- `make install`: Install project dependencies with Poetry
- `make install-dev`: Install development dependencies in conda environment
- `make test`: Run tests
- `make test-cov`: Run tests with coverage report
- `make lint`: Run linters (black, isort, mypy)
- `make format`: Format code with black and isort
- `make run-local-jupyter`: Start Jupyter Lab server
- `make clean`: Clean up temporary files

## Project Structure

- `stnf/`: Main package code
- `scripts/`: Training and analysis scripts
- `configs/`: Configuration files
- `data/`: Dataset files
- `envs/`: Environment setup scripts (conda and jupyter)
- `envs/.bin/`: Utility scripts

## Usage

### Training

```bash
python scripts/train_st_interp.py --config configs/config_st_interp.yaml
```

### Grid Search

```bash
python scripts/run_grid_search.py --config configs/config_st_interp.yaml --parallel --n_jobs 10
```

### Jupyter Lab

```bash
make run-local-jupyter
# or
bash envs/jupyter/start_jupyter_lab.sh
```

## Troubleshooting

### Conda not found

Make sure Conda is installed and initialized in your shell:
```bash
conda init bash  # or zsh
```

### Permission denied

Make shell scripts executable:
```bash
chmod +x envs/**/*.sh envs/.bin/*.sh
```
