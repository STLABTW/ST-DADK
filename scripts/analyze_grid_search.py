"""
Grid Search Results Analysis V3

Universal visualization framework:
- File structure: Separate plots per data file
- Subplot structure: Rows = obs_spatial_pattern, Columns = obs_ratio
- Within each subplot: Boxplots for all parameter combinations
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from itertools import product


def get_latest_grid_search_dir():
    """Find the most recently created grid_search directory in results folder"""
    results_path = Path('results')
    if not results_path.exists():
        return None
    
    grid_dirs = []
    for item in results_path.iterdir():
        if item.is_dir() and 'grid_search' in item.name:
            grid_dirs.append(item)
    
    if not grid_dirs:
        return None
    
    latest_dir = max(grid_dirs, key=lambda p: p.stat().st_mtime)
    return str(latest_dir)


def identify_varying_parameters(df_detail):
    """
    Identify which parameters vary across experiments
    (exclude fixed experimental setup parameters)
    """
    # Parameters that define experimental conditions (not model parameters)
    experimental_params = ['obs_method', 'obs_ratio', 'obs_spatial_pattern', 'data_file']
    
    # Check all columns for varying parameters
    varying_params = []
    param_values = {}
    
    for col in df_detail.columns:
        if col in ['config_id', 'tag', 'experiment_id', 'test_rmse', 'test_mae', 
                   'test_mse', 'valid_rmse', 'valid_mae', 'valid_mse',
                   'train_rmse', 'train_mae', 'train_mse', 'total_time_seconds']:
            continue
        
        if col in experimental_params:
            continue
        
        unique_vals = df_detail[col].unique()
        if len(unique_vals) > 1:
            varying_params.append(col)
            param_values[col] = sorted(unique_vals)
    
    return varying_params, param_values


def create_method_label(row, varying_params):
    """
    Create a label for a method based on varying parameters
    """
    label_parts = []
    
    # Define abbreviations for common parameters
    abbreviations = {
        'spatial_basis_function': {
            'wendland': 'Wend',
            'gaussian': 'Gaus',
            'triangular': 'Tria'
        },
        'spatial_init_method': {
            'uniform': 'Uni',
            'gmm': 'GMM',
            'random': 'Rand',
            'kmeans': 'KM'
        },
        'spatial_learnable': {
            True: 'Lrn',
            False: 'Fix'
        }
    }
    
    for param in varying_params:
        value = row[param]
        
        # Use abbreviation if available
        if param in abbreviations and value in abbreviations[param]:
            label_parts.append(abbreviations[param][value])
        else:
            # Fallback: use first 4 characters
            label_parts.append(str(value)[:4].capitalize())
    
    return '+'.join(label_parts)


def assign_colors(method_labels):
    """
    Assign colors to methods using a color palette
    """
    import matplotlib.pyplot as plt
    
    n_methods = len(method_labels)
    
    if n_methods <= 10:
        # Use tab10 for up to 10 methods
        cmap = plt.colormaps.get_cmap('tab10')
        colors = {label: cmap(i) for i, label in enumerate(method_labels)}
    else:
        # Use tab20 for more methods
        cmap = plt.colormaps.get_cmap('tab20')
        colors = {label: cmap(i % 20) for i, label in enumerate(method_labels)}
    
    return colors


def main():
    parser = argparse.ArgumentParser(description='Analyze grid search results (V3)')
    parser.add_argument('grid_dir', type=str, nargs='?', 
                        default=None,
                        help='Grid search directory path (default: latest)')
    args = parser.parse_args()
    
    # Determine grid directory
    if args.grid_dir is None:
        latest_dir = get_latest_grid_search_dir()
        if latest_dir is None:
            print("❌ No grid_search directories found in results/")
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
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {results_dir}")
    print(f"{'='*80}\n")
    
    df_detail = pd.read_csv(detail_csv)
    
    # Extract data_file from tag if not present
    if 'data_file' not in df_detail.columns:
        extracted = df_detail['tag'].str.extract(r'(data/\w+/\w+\.csv)')
        if extracted[0].notna().any():
            df_detail['data_file'] = extracted[0]
        else:
            # No data file pattern in tag - assign a default value
            df_detail['data_file'] = 'all_data'
    
    # Identify varying parameters
    varying_params, param_values = identify_varying_parameters(df_detail)
    
    print(f"📊 Identified varying parameters:")
    for param in varying_params:
        values = param_values[param]
        print(f"   - {param}: {values}")
    print()
    
    # Create method labels based on varying parameters
    df_detail['method'] = df_detail.apply(
        lambda row: create_method_label(row, varying_params), axis=1
    )
    
    # Get unique methods and assign colors
    methods = sorted(df_detail['method'].unique())
    method_colors = assign_colors(methods)
    
    print(f"🎨 Methods ({len(methods)}):")
    for method in methods:
        print(f"   - {method}")
    print()
    
    # Create output directory
    output_dir = results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Get unique data files
    data_files = sorted(df_detail['data_file'].dropna().unique())
    
    print(f"📁 Data files ({len(data_files)}):")
    for df_file in data_files:
        print(f"   - {df_file}")
    print()
    
    # ========================================================================
    # Process each data file
    # ========================================================================
    print(f"{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    for data_file in data_files:
        print(f"Processing: {data_file}")
        
        # Filter data for this file
        df_file = df_detail[df_detail['data_file'] == data_file]
        
        # Get unique values for subplot arrangement
        obs_patterns = sorted(df_file['obs_spatial_pattern'].unique())
        obs_ratios = sorted(df_file['obs_ratio'].unique())
        obs_methods = sorted(df_file['obs_method'].unique())
        
        # Create subplot grid: rows = patterns, cols = (method, ratio) combinations
        n_rows = len(obs_patterns)
        n_cols = len(obs_methods) * len(obs_ratios)
        
        if n_cols == 0 or n_rows == 0:
            print(f"  ⚠️  No data to plot")
            continue
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(6 * n_cols, 5 * n_rows),
                                squeeze=False)
        
        fig.suptitle(f'Performance Comparison: {data_file}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Generate column configurations
        col_configs = []
        for obs_method in obs_methods:
            for obs_ratio in obs_ratios:
                col_configs.append((obs_method, obs_ratio))
        
        # Plot each subplot
        for row_idx, pattern in enumerate(obs_patterns):
            for col_idx, (obs_method, obs_ratio) in enumerate(col_configs):
                ax = axes[row_idx, col_idx]
                
                # Filter data for this subplot
                df_subplot = df_file[
                    (df_file['obs_spatial_pattern'] == pattern) &
                    (df_file['obs_method'] == obs_method) &
                    (df_file['obs_ratio'] == obs_ratio)
                ]
                
                if len(df_subplot) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{pattern.capitalize()} | {obs_method}, r={obs_ratio}', 
                                fontsize=11, fontweight='bold')
                    continue
                
                # Prepare data for boxplot
                data_for_plot = []
                labels_for_plot = []
                colors_for_plot = []
                
                for method in methods:
                    df_method = df_subplot[df_subplot['method'] == method]
                    if len(df_method) > 0:
                        data_for_plot.append(df_method['test_rmse'].values)
                        labels_for_plot.append(method)
                        colors_for_plot.append(method_colors[method])
                
                if len(data_for_plot) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{pattern.capitalize()} | {obs_method}, r={obs_ratio}', 
                                fontsize=11, fontweight='bold')
                    continue
                
                # Create boxplot
                positions = np.arange(len(data_for_plot))
                bp = ax.boxplot(data_for_plot, positions=positions,
                               widths=0.6, patch_artist=True,
                               medianprops=dict(color='black', linewidth=2),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5),
                               flierprops=dict(marker='o', markersize=5, alpha=0.5))
                
                # Color boxes
                for patch, color in zip(bp['boxes'], colors_for_plot):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Set labels
                ax.set_xticks(positions)
                ax.set_xticklabels(labels_for_plot, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Test RMSE', fontsize=10, fontweight='bold')
                ax.set_title(f'{pattern.capitalize()} | {obs_method}, r={obs_ratio}', 
                            fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        
        # Save figure
        filename = data_file.replace('/', '_').replace('.csv', '.png')
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    summary_stats = []
    
    for data_file in data_files:
        df_file = df_detail[df_detail['data_file'] == data_file]
        
        obs_methods = df_file['obs_method'].unique()
        obs_ratios = df_file['obs_ratio'].unique()
        obs_patterns = df_file['obs_spatial_pattern'].unique()
        
        for obs_method in obs_methods:
            for obs_ratio in obs_ratios:
                for pattern in obs_patterns:
                    for method in methods:
                        df_sub = df_file[
                            (df_file['obs_method'] == obs_method) &
                            (df_file['obs_ratio'] == obs_ratio) &
                            (df_file['obs_spatial_pattern'] == pattern) &
                            (df_file['method'] == method)
                        ]
                        
                        if len(df_sub) > 0:
                            rmse_values = df_sub['test_rmse'].values
                            summary_stats.append({
                                'data_file': data_file,
                                'obs_method': obs_method,
                                'obs_ratio': obs_ratio,
                                'pattern': pattern,
                                'method': method,
                                'rmse_mean': np.mean(rmse_values),
                                'rmse_std': np.std(rmse_values),
                                'rmse_min': np.min(rmse_values),
                                'rmse_max': np.max(rmse_values),
                                'n_experiments': len(rmse_values)
                            })
    
    df_summary = pd.DataFrame(summary_stats)
    summary_file = output_dir / 'detailed_summary.csv'
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Summary saved: {summary_file}")
    
    # ========================================================================
    # Key Insights
    # ========================================================================
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")
    
    for data_file in data_files:
        df_file_summary = df_summary[df_summary['data_file'] == data_file]
        
        if len(df_file_summary) == 0:
            continue
        
        print(f"📊 {data_file}:")
        
        # Best method overall
        best_idx = df_file_summary['rmse_mean'].idxmin()
        best = df_file_summary.loc[best_idx]
        print(f"   ✅ Best overall: {best['method']}")
        print(f"      RMSE: {best['rmse_mean']:.4f} ± {best['rmse_std']:.4f}")
        print(f"      ({best['pattern']}, {best['obs_method']}, r={best['obs_ratio']})")
        
        # Best by pattern
        for pattern in df_file_summary['pattern'].unique():
            df_pattern = df_file_summary[df_file_summary['pattern'] == pattern]
            if len(df_pattern) > 0:
                best_idx = df_pattern['rmse_mean'].idxmin()
                best = df_pattern.loc[best_idx]
                print(f"   📌 Best for {pattern}: {best['method']} "
                      f"(RMSE: {best['rmse_mean']:.4f} ± {best['rmse_std']:.4f})")
        
        print()
    
    print(f"{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
