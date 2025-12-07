"""
Grid Search Results Analysis V2

Focused visualization: Method comparison by pattern, separated by data file
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def get_latest_grid_search_dir():
    """Find the most recently created grid_search directory in results folder"""
    results_path = Path('results')
    if not results_path.exists():
        return None
    
    # Find all directories containing 'grid_search' in their name
    grid_dirs = []
    for item in results_path.iterdir():
        if item.is_dir() and 'grid_search' in item.name:
            grid_dirs.append(item)
    
    if not grid_dirs:
        return None
    
    # Return the most recently modified directory
    latest_dir = max(grid_dirs, key=lambda p: p.stat().st_mtime)
    return str(latest_dir)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze grid search results')
parser.add_argument('grid_dir', type=str, nargs='?', 
                    default=None,
                    help='Grid search directory path (default: latest grid_search folder in results/)')
args = parser.parse_args()

# Determine grid directory
if args.grid_dir is None:
    latest_dir = get_latest_grid_search_dir()
    if latest_dir is None:
        print("❌ No grid_search directories found in results/")
        print("Please specify a grid_dir explicitly:")
        print("  python scripts/analyze_grid_search.py <grid_dir>")
        exit(1)
    results_dir = Path(latest_dir)
    print(f"Using latest grid_search directory: {results_dir}")
else:
    results_dir = Path(args.grid_dir)

# Load data
if not results_dir.exists():
    print(f"❌ Grid directory not found: {results_dir}")
    exit(1)

detail_csv = results_dir / 'grid_search_detail.csv'
if not detail_csv.exists():
    print(f"❌ grid_search_detail.csv not found in {results_dir}")
    exit(1)

print(f"Analyzing results from: {results_dir}")
df_detail = pd.read_csv(detail_csv)

# Extract data_file from tag
df_detail['data_file'] = df_detail['tag'].str.extract(r'(data/\w+/\w+\.csv)')

# Create method labels
def create_method_label(row):
    init = 'GMM' if row['spatial_init_method'] == 'gmm' else 'Uniform'
    learn = 'Lrn' if row['spatial_learnable'] else 'Fix'
    return f"{init}+{learn}"

df_detail['method'] = df_detail.apply(create_method_label, axis=1)

# Create output directory
output_dir = results_dir / 'analysis'
output_dir.mkdir(exist_ok=True)

# Color palette for methods
method_colors = {
    'Uniform+Fix': '#95a5a6',
    'Uniform+Lrn': '#3498db',
    'GMM+Fix': '#e67e22',
    'GMM+Lrn': '#e74c3c'
}

# methods = ['Uniform+Fix', 'Uniform+Lrn', 'GMM+Fix', 'GMM+Lrn']
methods = ['Uniform+Fix', 'GMM+Lrn']

# Get unique data files
data_files = sorted(df_detail['data_file'].unique())

print(f"Found {len(data_files)} data files")
print("="*80)

# Process each data file
for data_file in data_files:
    if pd.isna(data_file):
        continue
    
    print(f"\nProcessing: {data_file}")
    
    # Filter data for this file
    df_file = df_detail[df_detail['data_file'] == data_file]
    
    # Get unique obs_methods, obs_ratios, and patterns
    unique_obs_methods = sorted(df_file['obs_method'].unique())
    unique_obs_ratios = sorted(df_file['obs_ratio'].unique())
    unique_patterns = sorted(df_file['obs_spatial_pattern'].unique())
    
    # Create subplot configurations dynamically
    configs = []
    for obs_method in unique_obs_methods:
        for obs_ratio in unique_obs_ratios:
            configs.append((obs_method, obs_ratio, len(configs)))
    
    n_cols = len(configs)
    n_rows = len(unique_patterns)
    
    # Create figure with n_rows×n_cols subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Ensure axes is always 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Performance Comparison: {data_file}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Define subplot configurations
    # configs = [
    #     ('site-wise', 0.25, 0),
    #     ('site-wise', 0.5, 1),
    #     ('random', 0.25, 2),
    #     ('random', 0.5, 3)
    # ]
    
    for obs_method, obs_ratio, col_idx in configs:
        # Create mapping for patterns to row indices
        pattern_to_row = {pattern: idx for idx, pattern in enumerate(unique_patterns)}
        
        # Filter data for this configuration
        df_config = df_file[
            (df_file['obs_method'] == obs_method) &
            (df_file['obs_ratio'] == obs_ratio)
        ]
        
        if len(df_config) == 0:
            # Mark all pattern rows as no data
            for pattern in unique_patterns:
                row_idx = pattern_to_row[pattern]
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{pattern.capitalize()} | {obs_method}, ratio={obs_ratio}', 
                            fontsize=11, fontweight='bold')
            continue
        
        # Prepare data for each pattern
        for pattern in unique_patterns:
            row_idx = pattern_to_row[pattern]
            ax = axes[row_idx, col_idx]
            
            data_for_pattern = []
            
            for method in methods:
                df_method = df_config[
                    (df_config['method'] == method) &
                    (df_config['obs_spatial_pattern'] == pattern)
                ]
                
                data_for_pattern.append(df_method['test_rmse'].values if len(df_method) > 0 else [])
            
            # Choose color based on pattern
            if pattern == 'uniform':
                box_color = '#3498db'
                median_color = 'darkblue'
            else:  # corner
                box_color = '#e74c3c'
                median_color = 'darkred'
            
            # Plot
            bp = ax.boxplot(data_for_pattern, positions=range(len(methods)),
                           widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor=box_color, alpha=0.7),
                           medianprops=dict(color=median_color, linewidth=2),
                           whiskerprops=dict(color=box_color),
                           capprops=dict(color=box_color),
                           flierprops=dict(marker='o', markerfacecolor=box_color, 
                                         markersize=5, alpha=0.5))
            
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
            ax.set_ylabel('Test RMSE', fontsize=10, fontweight='bold')
            ax.set_title(f'{pattern.capitalize()} | {obs_method}, ratio={obs_ratio}', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    filename = data_file.replace('/', '_').replace('.csv', '.png')
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

summary_stats = []

# Get unique values from data
unique_obs_methods = df_detail['obs_method'].unique()
unique_obs_ratios = df_detail['obs_ratio'].unique()
unique_patterns = df_detail['obs_spatial_pattern'].unique()

for data_file in data_files:
    if pd.isna(data_file):
        continue
    
    df_file = df_detail[df_detail['data_file'] == data_file]
    
    for obs_method in unique_obs_methods:
        for obs_ratio in unique_obs_ratios:
            for method in methods:
                for pattern in unique_patterns:
                    df_sub = df_file[
                        (df_file['obs_method'] == obs_method) &
                        (df_file['obs_ratio'] == obs_ratio) &
                        (df_file['method'] == method) &
                        (df_file['obs_spatial_pattern'] == pattern)
                    ]
                    
                    if len(df_sub) > 0:
                        rmse_values = df_sub['test_rmse'].values
                        summary_stats.append({
                            'data_file': data_file,
                            'obs_method': obs_method,
                            'obs_ratio': obs_ratio,
                            'method': method,
                            'pattern': pattern,
                            'rmse_mean': np.mean(rmse_values),
                            'rmse_std': np.std(rmse_values),
                            'rmse_min': np.min(rmse_values),
                            'rmse_max': np.max(rmse_values),
                            'n_experiments': len(rmse_values)
                        })

df_summary = pd.DataFrame(summary_stats)
df_summary.to_csv(output_dir / 'detailed_summary.csv', index=False)
print(f"\nSaved: {output_dir / 'detailed_summary.csv'}")

# Print key insights
print("\n" + "="*80)
print("KEY INSIGHTS BY DATA FILE")
print("="*80)

# Check if data_file column exists in df_summary
if 'data_file' in df_summary.columns:
    for data_file in data_files:
        if pd.isna(data_file):
            continue
        
        print(f"\n{data_file}:")
        df_file_summary = df_summary[df_summary['data_file'] == data_file]
        
        # Best method for each pattern
        for pattern in ['uniform', 'corner']:
            df_pattern = df_file_summary[df_file_summary['pattern'] == pattern]
            if len(df_pattern) > 0:
                best_idx = df_pattern['rmse_mean'].idxmin()
                best = df_pattern.loc[best_idx]
                print(f"  Best for {pattern}: {best['method']} "
                      f"(RMSE: {best['rmse_mean']:.4f} ± {best['rmse_std']:.4f})")
        
        # Pattern effect by method
        print(f"  Pattern effect (Corner - Uniform):")
        for method in methods:
            df_method = df_file_summary[df_file_summary['method'] == method]
            if len(df_method) > 0:
                corner_mean = df_method[df_method['pattern'] == 'corner']['rmse_mean'].mean()
                uniform_mean = df_method[df_method['pattern'] == 'uniform']['rmse_mean'].mean()
                if not (np.isnan(corner_mean) or np.isnan(uniform_mean)):
                    diff = corner_mean - uniform_mean
                    pct = (diff / uniform_mean) * 100
                    print(f"    {method}: +{diff:.4f} ({pct:+.1f}%)")
else:
    print("\nSummary statistics across all data:")
    
    # Best method for each pattern
    for pattern in ['uniform', 'corner']:
        df_pattern = df_summary[df_summary['pattern'] == pattern]
        if len(df_pattern) > 0:
            best_idx = df_pattern['rmse_mean'].idxmin()
            best = df_pattern.loc[best_idx]
            print(f"  Best for {pattern}: {best['method']} "
                  f"(RMSE: {best['rmse_mean']:.4f} ± {best['rmse_std']:.4f})")
    
    # Pattern effect by method
    print(f"\n  Pattern effect (Corner - Uniform):")
    for method in methods:
        df_method = df_summary[df_summary['method'] == method]
        if len(df_method) > 0:
            corner_mean = df_method[df_method['pattern'] == 'corner']['rmse_mean'].mean()
            uniform_mean = df_method[df_method['pattern'] == 'uniform']['rmse_mean'].mean()
            if not (np.isnan(corner_mean) or np.isnan(uniform_mean)):
                diff = corner_mean - uniform_mean
                pct = (diff / uniform_mean) * 100
                print(f"    {method}: +{diff:.4f} ({pct:+.1f}%)")

print("\n" + "="*80)
print("Analysis complete! Check the 'analysis' folder for visualizations.")
print("="*80)
