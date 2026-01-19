# Grid Search Experiment System User Guide

## Overview

A system that automatically performs experiments for all hyperparameter combinations and saves results in structured CSV files.

## Structural Changes

### Previous Approach
- Created 32 individual config files (config_1.yaml ~ config_32.yaml)
- Executed each config sequentially
- Results stored in separate folders
- Manual result collection and organization required

### New Approach
- **Single base config file** (config_st_interp.yaml)
- **Parameter grid definition** (in run_grid_search.py)
- **Automatic combination generation and execution**
- **Unified CSV file output**:
  - `grid_search_summary.csv`: Summary statistics for each config
  - `grid_search_detail.csv`: Raw values for each iteration
  - `grid_search_configs.csv`: Complete config information

## File Structure

```
ST-DADK/
├── configs/
│   └── config_st_interp.yaml          # Base configuration
├── scripts/
│   ├── train_st_interp.py             # Existing training script (modified)
│   └── run_grid_search.py             # New Grid Search runner
└── results/
    └── YYYYMMDD_grid_search/
        ├── grid_search_summary.csv     # Summary statistics
        ├── grid_search_detail.csv      # Detailed results
        ├── grid_search_configs.csv     # Config information
        └── config001_uni_lrn_site_10_cor/
            ├── config.yaml
            ├── experiments/
            │   ├── 1/
            │   ├── 2/
            │   └── ...
            └── summary/
                ├── summary_statistics.json
                └── all_experiments.csv
```

## Usage

### 1. Define Parameter Grid

Define the parameter combinations you want to experiment with in `scripts/run_grid_search.py`:

```python
param_grid = {
    'spatial_init_method': ['uniform', 'gmm'],
    'spatial_learnable': [True, False],
    'obs_method': ['site-wise', 'random'],
    'obs_ratio': [0.1, 0.3],
    'obs_spatial_pattern': ['corner', 'uniform'],
}
```

### 2. Run Experiments

**Sequential execution:**
```bash
python scripts/run_grid_search.py --config configs/config_st_interp.yaml
```

**Parallel execution (recommended):**
```bash
python scripts/run_grid_search.py \
    --config configs/config_st_interp.yaml \
    --parallel \
    --n_jobs 10
```

**Specify output directory:**
```bash
python scripts/run_grid_search.py \
    --config configs/config_st_interp.yaml \
    --output_dir results/my_experiment \
    --parallel \
    --n_jobs 10
```

### 3. Analyze Results

#### Summary CSV (grid_search_summary.csv)

Average performance for each config:

```python
import pandas as pd

df = pd.read_csv('results/20251203_grid_search/grid_search_summary.csv')

# Best performers
df_sorted = df.sort_values('test_rmse_mean')
print(df_sorted.head(10))

# Factor analysis
for method in df['spatial_init_method'].unique():
    subset = df[df['spatial_init_method'] == method]
    print(f"{method}: {subset['test_rmse_mean'].mean():.4f}")
```

#### Detail CSV (grid_search_detail.csv)

Raw values for each iteration:

```python
df_detail = pd.read_csv('results/20251203_grid_search/grid_search_detail.csv')

# Specific config analysis
config1 = df_detail[df_detail['config_id'] == 1]
print(config1[['experiment_id', 'test_rmse', 'test_mae']])

# Statistical tests
from scipy import stats
group1 = df_detail[df_detail['config_id'] == 1]['test_rmse']
group2 = df_detail[df_detail['config_id'] == 2]['test_rmse']
t_stat, p_value = stats.ttest_ind(group1, group2)
```

## Output File Details

### 1. grid_search_summary.csv

One row per config, including summary statistics:

| Column | Description |
|--------|-------------|
| config_id | Config number (1-32) |
| tag | Config identifier |
| spatial_init_method | 'uniform' or 'gmm' |
| spatial_learnable | True or False |
| obs_method | 'site-wise' or 'random' |
| obs_ratio | 0.1 or 0.3 |
| obs_spatial_pattern | 'corner' or 'uniform' |
| n_experiments | Number of repeated experiments |
| test_rmse_mean | Test RMSE mean |
| test_rmse_std | Test RMSE standard deviation |
| test_rmse_min | Test RMSE minimum |
| test_rmse_max | Test RMSE maximum |
| test_rmse_median | Test RMSE median |
| ... | (same for other metrics) |

### 2. grid_search_detail.csv

One row per iteration for each config:

| Column | Description |
|--------|-------------|
| config_id | Config number |
| tag | Config identifier |
| experiment_id | Iteration number (1-10) |
| spatial_init_method | Config setting |
| ... | (other config settings) |
| test_rmse | Test RMSE for this iteration |
| test_mae | Test MAE for this iteration |
| test_mse | Test MSE for this iteration |
| valid_rmse | Valid RMSE for this iteration |
| ... | (other metrics) |
| total_time_seconds | Training time (seconds) |

### 3. grid_search_configs.csv

Stores complete config information (JSON format):

| Column | Description |
|--------|-------------|
| config_id | Config number |
| tag | Config identifier |
| config_json | Complete config (JSON string) |

## Visualization Examples

### 1. Heatmap Comparison

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results/20251203_grid_search/grid_search_summary.csv')

# Pivot for heatmap
pivot = df.pivot_table(
    values='test_rmse_mean',
    index=['spatial_init_method', 'spatial_learnable'],
    columns=['obs_method', 'obs_ratio', 'obs_spatial_pattern']
)

plt.figure(figsize=(15, 6))
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r')
plt.title('Test RMSE across all configurations')
plt.tight_layout()
plt.savefig('heatmap.png')
```

### 2. Factor Effect Comparison

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/20251203_grid_search/grid_search_summary.csv')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

factors = ['spatial_init_method', 'spatial_learnable', 
           'obs_method', 'obs_ratio', 'obs_spatial_pattern']

for ax, factor in zip(axes.flatten(), factors):
    data = df.groupby(factor)['test_rmse_mean'].agg(['mean', 'std'])
    data.plot(kind='bar', y='mean', yerr='std', ax=ax)
    ax.set_title(f'Effect of {factor}')
    ax.set_ylabel('Test RMSE')

plt.tight_layout()
plt.savefig('factor_effects.png')
```

### 3. Interaction Plot

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/20251203_grid_search/grid_search_summary.csv')

# uni_fix vs gmm_lrn by obs_spatial_pattern
df_unifix = df[(df['spatial_init_method']=='uniform') & (df['spatial_learnable']==False)]
df_gmmlrn = df[(df['spatial_init_method']=='gmm') & (df['spatial_learnable']==True)]

patterns = df['obs_spatial_pattern'].unique()
unifix_means = [df_unifix[df_unifix['obs_spatial_pattern']==p]['test_rmse_mean'].mean() 
                for p in patterns]
gmmlrn_means = [df_gmmlrn[df_gmmlrn['obs_spatial_pattern']==p]['test_rmse_mean'].mean() 
                for p in patterns]

plt.figure(figsize=(10, 6))
plt.plot(patterns, unifix_means, 'o-', label='uni_fix', linewidth=2)
plt.plot(patterns, gmmlrn_means, 's-', label='gmm_lrn', linewidth=2)
plt.xlabel('Observation Pattern')
plt.ylabel('Test RMSE')
plt.title('Interaction: Method × Observation Pattern')
plt.legend()
plt.grid(True)
plt.savefig('interaction_plot.png')
```

## Advantages

1. **Reproducibility**: All experiments start from the same base config
2. **Scalability**: Easy to add parameters (just modify param_grid)
3. **Easy Analysis**: Various analyses possible with structured CSV
4. **Parallelization**: All configs can be executed simultaneously
5. **Traceability**: All configs and results are automatically saved

## Experimenting with Additional Parameters

To add other parameters, modify `run_grid_search.py`:

```python
param_grid = {
    # Existing parameters
    'spatial_init_method': ['uniform', 'gmm'],
    'spatial_learnable': [True, False],
    
    # Add new parameters
    'lr': [1e-3, 5e-3, 1e-2],
    'hidden_dims': [[128, 128], [256, 256, 128]],
    'dropout': [0.0, 0.1, 0.2],
}
```

## Troubleshooting

### Out of Memory
```bash
# Reduce n_jobs
python scripts/run_grid_search.py --parallel --n_jobs 5
```

### Re-run Specific Config Only
```python
# Run directly in Python
from scripts.train_st_interp import run_multiple_experiments
import yaml

with open('configs/config_st_interp.yaml') as f:
    config = yaml.safe_load(f)

config['spatial_init_method'] = 'gmm'
config['obs_ratio'] = 0.3
# ... other settings

summary = run_multiple_experiments(config, 'results/rerun', 'cpu', parallel=True)
```
