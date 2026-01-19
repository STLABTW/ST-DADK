# Scripts Usage Guide

This directory contains all executable scripts for training, grid search, analysis, and visualization.

## Table of Contents

- [Training Scripts](#training-scripts)
- [Grid Search Scripts](#grid-search-scripts)
- [Analysis Scripts](#analysis-scripts)
- [Visualization Scripts](#visualization-scripts)

## Training Scripts

### `train_st_interp.py`

Train a single spatio-temporal interpolation model.

**Basic Usage:**
```bash
python scripts/train_st_interp.py --config configs/config_st_interp.yaml
```

**Arguments:**
- `--config` (str, default: `configs/config_st_interp.yaml`): Path to configuration YAML file
- `--data_file` (str, optional): Override data file path from config
- `--n_experiments` (int, optional): Override number of experiments from config
- `--base_seed` (int, optional): Override base seed from config
- `--parallel` (flag): Run multiple experiments in parallel
- `--n_jobs` (int, default: -1): Number of parallel jobs (-1 for all CPUs, 0 for sequential)
- `--start_exp_id` (int, optional): Starting experiment ID (1-based)
- `--end_exp_id` (int, optional): Ending experiment ID (inclusive)
- `--skip-existing` (flag): Skip experiments that already have results.json

**Examples:**
```bash
# Train with default config
python scripts/train_st_interp.py --config configs/config_st_interp.yaml

# Train with custom data file
python scripts/train_st_interp.py --config configs/config_st_interp.yaml --data_file data/2a/2a_8.csv

# Run 10 experiments in parallel
python scripts/train_st_interp.py --config configs/config_st_interp.yaml --n_experiments 10 --parallel --n_jobs 4

# Resume experiments 5-10
python scripts/train_st_interp.py --config configs/config_st_interp.yaml --start_exp_id 5 --end_exp_id 10 --skip-existing
```

## Grid Search Scripts

### `run_grid_search.py`

Run grid search experiments across multiple hyperparameter combinations.

**Basic Usage:**
```bash
python scripts/run_grid_search.py --config configs/config_st_interp.yaml
```

**Arguments:**
- `--config` (str, default: `configs/config_st_interp.yaml`): Base configuration file
- `--output_dir` (str, optional): Output directory (default: `results/<date>_grid_search`)
- `--parallel` (flag): Run experiments in parallel
- `--n_jobs` (int, default: 10): Number of parallel jobs

**Examples:**
```bash
# Sequential execution
python scripts/run_grid_search.py --config configs/config_st_interp.yaml

# Parallel execution (recommended)
python scripts/run_grid_search.py --config configs/config_st_interp.yaml --parallel --n_jobs 10

# Custom output directory
python scripts/run_grid_search.py --config configs/config_st_interp.yaml --output_dir results/my_experiment --parallel --n_jobs 10
```

**Note:** The parameter grid is defined inside the script. Edit `param_grid` in `run_grid_search.py` to customize hyperparameter combinations.

### `resume_grid_search.py`

Resume or continue grid search experiments for specific experiment IDs.

**Basic Usage:**
```bash
python scripts/resume_grid_search.py --grid_dir results/YYYYMMDD_grid_search --start_exp_id 17 --end_exp_id 20
```

**Arguments:**
- `--grid_dir` (str, required): Grid search directory (e.g., `results/20251206_224336_grid_search`)
- `--start_exp_id` (int, optional): Starting experiment ID (required unless `--summarize-only`)
- `--end_exp_id` (int, optional): Ending experiment ID (required unless `--summarize-only`)
- `--config_filter` (str, optional): Only process configs matching this pattern (e.g., `"config001"`)
- `--summarize-only` (flag): Only regenerate summaries from existing results, do not run experiments
- `--skip-existing` (flag): Skip experiments that already have results.json

**Examples:**
```bash
# Resume experiments 17-20 for all configs
python scripts/resume_grid_search.py --grid_dir results/20251206_224336_grid_search --start_exp_id 17 --end_exp_id 20

# Resume for specific config only
python scripts/resume_grid_search.py --grid_dir results/20251206_224336_grid_search --start_exp_id 17 --end_exp_id 20 --config_filter config001

# Only regenerate summaries without running experiments
python scripts/resume_grid_search.py --grid_dir results/20251206_224336_grid_search --summarize-only

# Skip existing experiments
python scripts/resume_grid_search.py --grid_dir results/20251206_224336_grid_search --start_exp_id 17 --end_exp_id 20 --skip-existing
```

## Analysis Scripts

### `analyze_grid_search.py`

Analyze and visualize grid search results.

**Basic Usage:**
```bash
python scripts/analyze_grid_search.py results/YYYYMMDD_grid_search
```

**Arguments:**
- `grid_dir` (str, positional): Grid search results directory
- `--summarize-only` (flag): Only regenerate summary files, skip visualization

**Examples:**
```bash
# Full analysis with visualizations
python scripts/analyze_grid_search.py results/20251206_224336_grid_search

# Only regenerate summary files
python scripts/analyze_grid_search.py results/20251206_224336_grid_search --summarize-only
```

**Output:**
- Regenerates `grid_search_summary.csv`, `grid_search_detail.csv`, and `grid_search_configs.json`
- Creates visualization plots in `results/YYYYMMDD_grid_search/analysis/`
- Generates detailed summary statistics

## Visualization Scripts

### `visualize_2b_data.py`

Visualize 2b dataset characteristics.

**Usage:**
```bash
python scripts/visualize_2b_data.py
```

### `visualize_obs_density.py`

Visualize observation density patterns.

**Usage:**
```bash
python scripts/visualize_obs_density.py
```

## Common Workflows

### 1. Single Model Training

```bash
# Train a single model with default config
python scripts/train_st_interp.py --config configs/config_st_interp.yaml
```

### 2. Grid Search Workflow

```bash
# Step 1: Run grid search
python scripts/run_grid_search.py --config configs/config_st_interp.yaml --parallel --n_jobs 10

# Step 2: Analyze results
python scripts/analyze_grid_search.py results/YYYYMMDD_grid_search

# Step 3: If needed, resume incomplete experiments
python scripts/resume_grid_search.py --grid_dir results/YYYYMMDD_grid_search --start_exp_id 17 --end_exp_id 20
```

### 3. Resume Failed Experiments

```bash
# Resume specific experiment range
python scripts/resume_grid_search.py \
    --grid_dir results/YYYYMMDD_grid_search \
    --start_exp_id 17 \
    --end_exp_id 20 \
    --skip-existing
```

### 4. Regenerate Summaries

```bash
# Regenerate summaries from existing results
python scripts/resume_grid_search.py --grid_dir results/YYYYMMDD_grid_search --summarize-only

# Or use analyze script
python scripts/analyze_grid_search.py results/YYYYMMDD_grid_search --summarize-only
```

## Configuration Files

All scripts use YAML configuration files located in `configs/`:
- `config_st_interp.yaml`: Main configuration file with all hyperparameters

Key configuration sections:
- Model architecture (basis functions, centers, etc.)
- Training settings (epochs, learning rate, batch size, etc.)
- Data settings (observation method, ratio, pattern, etc.)
- Experiment settings (number of experiments, seeds, etc.)

## Output Structure

### Training Output
```
results/
└── <tag>/
    ├── config.yaml
    ├── experiments/
    │   ├── 1/
    │   │   ├── results.json
    │   │   └── ...
    │   └── ...
    └── summary/
        ├── summary_statistics.json
        └── all_experiments.csv
```

### Grid Search Output
```
results/
└── YYYYMMDD_grid_search/
    ├── grid_search_summary.csv
    ├── grid_search_detail.csv
    ├── grid_search_configs.json
    ├── config001_.../
    │   └── ...
    └── analysis/
        └── *.png
```

## Tips

1. **Parallel Execution**: Use `--parallel` for faster execution, but monitor memory usage
2. **Resume Experiments**: Use `--skip-existing` to avoid re-running completed experiments
3. **Custom Configs**: Create new YAML files in `configs/` for different experiment setups
4. **Parameter Grid**: Edit `param_grid` in `run_grid_search.py` to customize hyperparameter search space
5. **Analysis**: Run analysis after grid search completes to generate visualizations and insights
