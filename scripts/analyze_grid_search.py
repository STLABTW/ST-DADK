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
import json
import yaml


def regenerate_top_level_summaries(results_dir):
    """
    Regenerate top-level grid_search_summary.csv and grid_search_detail.csv
    based on per-config summary files
    
    Args:
        results_dir: Path to grid search results directory
    """
    results_path = Path(results_dir)
    
    # Find all config directories with summary files
    config_yaml_paths = list(results_path.rglob('config.yaml'))
    
    if not config_yaml_paths:
        print("❌ No config.yaml files found")
        return
    
    # Get directories containing config.yaml and summary/
    target_dirs = []
    for config_path in config_yaml_paths:
        config_dir = config_path.parent
        summary_dir = config_dir / 'summary'
        if summary_dir.exists() and (summary_dir / 'summary_statistics.json').exists():
            target_dirs.append(config_dir)
    
    target_dirs = sorted(target_dirs)
    
    if not target_dirs:
        print("❌ No valid config directories with summary files found")
        return
    
    print(f"Regenerating top-level summaries from {len(target_dirs)} configs...")
    
    # Collect all results
    all_configs = []
    summary_records = []
    detail_records = []
    
    for config_dir in target_dirs:
        # Load config
        with open(config_dir / 'config.yaml', 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Load summary statistics
        summary_file = config_dir / 'summary' / 'summary_statistics.json'
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        all_configs.append(config)
        
        # Build summary record
        record = {
            'config_id': config.get('config_id', 0),
            'tag': config.get('tag', ''),
            'spatial_basis_function': config.get('spatial_basis_function', 'wendland'),
            'spatial_init_method': config.get('spatial_init_method', ''),
            'spatial_learnable': config.get('spatial_learnable', False),
            'obs_method': config.get('obs_method', ''),
            'obs_ratio': config.get('obs_ratio', 0.0),
            'obs_spatial_pattern': config.get('obs_spatial_pattern', ''),
            'n_experiments': summary.get('n_experiments', 0),
        }
        
        # Add metric statistics - check if they're nested under 'statistics'
        stats_dict = summary.get('statistics', summary)  # Try 'statistics' key first, fallback to root
        
        for metric in ['test_rmse', 'test_mae', 'test_mse', 
                       'valid_rmse', 'valid_mae', 'valid_mse',
                       'train_rmse', 'train_mae', 'train_mse',
                       'total_time_seconds']:
            if metric in stats_dict:
                stats = stats_dict[metric]
                record[f'{metric}_mean'] = stats['mean']
                record[f'{metric}_std'] = stats['std']
                record[f'{metric}_min'] = stats['min']
                record[f'{metric}_max'] = stats['max']
                record[f'{metric}_median'] = stats['median']
        
        summary_records.append(record)
        
        # Load detailed results for grid_search_detail.csv
        all_exp_file = config_dir / 'summary' / 'all_experiments.csv'
        if all_exp_file.exists():
            df_exp = pd.read_csv(all_exp_file)
            for _, row in df_exp.iterrows():
                detail_record = {
                    'config_id': config.get('config_id', 0),
                    'tag': config.get('tag', ''),
                    'experiment_id': row['experiment_id'],
                    'spatial_basis_function': config.get('spatial_basis_function', 'wendland'),
                    'spatial_init_method': config.get('spatial_init_method', ''),
                    'spatial_learnable': config.get('spatial_learnable', False),
                    'obs_method': config.get('obs_method', ''),
                    'obs_ratio': config.get('obs_ratio', 0.0),
                    'obs_spatial_pattern': config.get('obs_spatial_pattern', ''),
                }
                
                # Add metrics
                for metric in ['test_rmse', 'test_mae', 'test_mse',
                               'valid_rmse', 'valid_mae', 'valid_mse',
                               'train_rmse', 'train_mae', 'train_mse',
                               'total_time_seconds']:
                    if metric in row:
                        detail_record[metric] = row[metric]
                
                detail_records.append(detail_record)
    
    # Save grid_search_summary.csv
    df_summary = pd.DataFrame(summary_records)
    df_summary = df_summary.sort_values('config_id')
    summary_file = results_path / 'grid_search_summary.csv'
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Updated: grid_search_summary.csv ({len(df_summary)} configs)")
    
    # Save grid_search_detail.csv
    df_detail = pd.DataFrame(detail_records)
    df_detail = df_detail.sort_values(['config_id', 'experiment_id'])
    detail_file = results_path / 'grid_search_detail.csv'
    df_detail.to_csv(detail_file, index=False)
    print(f"✓ Updated: grid_search_detail.csv ({len(df_detail)} experiments)")
    
    # Save grid_search_configs.json
    configs_dict = {str(cfg.get('config_id', 0)): cfg for cfg in all_configs}
    config_json_file = results_path / 'grid_search_configs.json'
    with open(config_json_file, 'w', encoding='utf-8') as f:
        json.dump(configs_dict, f, indent=2, ensure_ascii=False)
    print(f"✓ Updated: grid_search_configs.json ({len(configs_dict)} configs)")
    
    # Save grid_search_configs.csv (simple index)
    config_records = [{'config_id': cfg.get('config_id', 0), 'tag': cfg.get('tag', '')} 
                      for cfg in all_configs]
    df_configs = pd.DataFrame(config_records)
    df_configs = df_configs.sort_values('config_id')
    config_file = results_path / 'grid_search_configs.csv'
    df_configs.to_csv(config_file, index=False)
    print(f"✓ Updated: grid_search_configs.csv ({len(df_configs)} configs)\n")


def regenerate_config_summaries(results_dir):
    """
    Regenerate summary files for each config based on existing results.json files
    
    Args:
        results_dir: Path to grid search results directory
    """
    results_path = Path(results_dir)
    
    # Find all directories with config.yaml (may be nested)
    config_yaml_paths = list(results_path.rglob('config.yaml'))
    
    if not config_yaml_paths:
        print("❌ No config.yaml files found")
        return
    
    # Get directories containing config.yaml and experiments/
    target_dirs = []
    for config_path in config_yaml_paths:
        config_dir = config_path.parent
        experiments_dir = config_dir / 'experiments'
        if experiments_dir.exists():
            target_dirs.append(config_dir)
    
    target_dirs = sorted(target_dirs)
    
    if not target_dirs:
        print("❌ No valid config directories with experiments/ found")
        return
    
    print(f"Found {len(target_dirs)} config directories with experiments\n")
    
    success_count = 0
    failed_count = 0
    
    for config_dir in target_dirs:
        # Get relative path for display
        rel_path = config_dir.relative_to(results_path)
        print(f"Processing {rel_path}...", end=' ')
        
        try:
            experiments_dir = config_dir / 'experiments'
            
            # Collect all results from individual experiment directories
            all_results = []
            
            exp_subdirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])
            
            for exp_subdir in exp_subdirs:
                result_file = exp_subdir / 'results.json'
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                        all_results.append(result)
            
            if not all_results:
                print("❌ No results.json files found")
                failed_count += 1
                continue
            
            # Create summary directory
            summary_dir = config_dir / 'summary'
            summary_dir.mkdir(exist_ok=True)
            
            # Generate summary statistics
            n_experiments = len(all_results)
            
            # Extract metrics from all experiments
            metrics_data = {
                'train_mse': [],
                'train_mae': [],
                'train_rmse': [],
                'valid_mse': [],
                'valid_mae': [],
                'valid_rmse': [],
                'test_mse': [],
                'test_mae': [],
                'test_rmse': [],
                'total_time_seconds': []
            }
            
            exp_records = []
            
            for result in all_results:
                # Handle both old and new format
                if 'metrics' in result:
                    train_metrics = result['metrics']['train']
                    valid_metrics = result['metrics']['valid']
                    test_metrics = result['metrics']['test']
                else:
                    # Old format with direct keys
                    train_metrics = {'mse': result.get('train_mse', 0), 
                                   'mae': result.get('train_mae', 0),
                                   'rmse': result.get('train_rmse', 0)}
                    valid_metrics = {'mse': result.get('valid_mse', 0),
                                   'mae': result.get('valid_mae', 0),
                                   'rmse': result.get('valid_rmse', 0)}
                    test_metrics = {'mse': result.get('test_mse', 0),
                                  'mae': result.get('test_mae', 0),
                                  'rmse': result.get('test_rmse', 0)}
                
                metrics_data['train_mse'].append(train_metrics['mse'])
                metrics_data['train_mae'].append(train_metrics['mae'])
                metrics_data['train_rmse'].append(train_metrics['rmse'])
                metrics_data['valid_mse'].append(valid_metrics['mse'])
                metrics_data['valid_mae'].append(valid_metrics['mae'])
                metrics_data['valid_rmse'].append(valid_metrics['rmse'])
                metrics_data['test_mse'].append(test_metrics['mse'])
                metrics_data['test_mae'].append(test_metrics['mae'])
                metrics_data['test_rmse'].append(test_metrics['rmse'])
                metrics_data['total_time_seconds'].append(result.get('total_time_seconds', 0))
                
                # Record for CSV
                exp_records.append({
                    'experiment_id': result.get('experiment_id', 0),
                    'experiment_seed': result.get('experiment_seed', 0),
                    'train_mse': train_metrics['mse'],
                    'train_mae': train_metrics['mae'],
                    'train_rmse': train_metrics['rmse'],
                    'valid_mse': valid_metrics['mse'],
                    'valid_mae': valid_metrics['mae'],
                    'valid_rmse': valid_metrics['rmse'],
                    'test_mse': test_metrics['mse'],
                    'test_mae': test_metrics['mae'],
                    'test_rmse': test_metrics['rmse'],
                    'total_time_seconds': result.get('total_time_seconds', 0)
                })
            
            # Compute statistics
            summary = {
                'n_experiments': n_experiments,
                'statistics': {}
            }
            
            for metric_name, values in metrics_data.items():
                values_array = np.array(values)
                summary['statistics'][metric_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'median': float(np.median(values_array)),
                    'values': [float(v) for v in values]
                }
            
            # Save summary JSON
            summary_file = summary_dir / 'summary_statistics.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save detailed CSV
            df = pd.DataFrame(exp_records)
            csv_file = summary_dir / 'all_experiments.csv'
            df.to_csv(csv_file, index=False)
            
            print(f"✓ ({n_experiments} experiments)")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            failed_count += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Summary regeneration complete:")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"{'='*80}")


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
    parser.add_argument('--summarize-only', action='store_true',
                        help='Only regenerate per-config summaries without re-running analysis')
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
    
    # ========================================================================
    # Regenerate per-config summaries (always do this first)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 1: REGENERATING PER-CONFIG SUMMARIES")
    print(f"{'='*80}\n")
    
    regenerate_config_summaries(results_dir)
    
    # ========================================================================
    # Regenerate top-level summary files (grid_search_summary.csv, etc.)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 2: REGENERATING TOP-LEVEL SUMMARIES")
    print(f"{'='*80}\n")
    
    regenerate_top_level_summaries(results_dir)
    
    if args.summarize_only:
        print(f"\n{'='*80}")
        print("✅ SUMMARY REGENERATION COMPLETE!")
        print(f"{'='*80}\n")
        return
    
    # ========================================================================
    # Generate analysis plots
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 3: GENERATING ANALYSIS PLOTS")
    print(f"{'='*80}\n")
    
    detail_csv = results_dir / 'grid_search_detail.csv'
    if not detail_csv.exists():
        print(f"❌ grid_search_detail.csv not found in {results_dir}")
        exit(1)
    
    print(f"Analyzing: {results_dir}\n")
    
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
        # Generate aggregated plots (one parameter ignored at a time)
        # ========================================================================
        print(f"  📊 Generating aggregated plots...")
        
        for param_to_ignore in varying_params:
            # Create aggregated method labels (without the ignored parameter)
            remaining_params = [p for p in varying_params if p != param_to_ignore]
            
            if len(remaining_params) == 0:
                continue
            
            df_file['agg_method'] = df_file.apply(
                lambda row: create_method_label(row, remaining_params), axis=1
            )
            
            agg_methods = sorted(df_file['agg_method'].unique())
            agg_colors = assign_colors(agg_methods)
            
            # Create figure
            fig_agg, axes_agg = plt.subplots(n_rows, n_cols, 
                                            figsize=(6 * n_cols, 5 * n_rows),
                                            squeeze=False)
            
            fig_agg.suptitle(f'Performance Comparison: {data_file}\n(Aggregated over {param_to_ignore})', 
                            fontsize=16, fontweight='bold', y=0.995)
            
            # Plot each subplot
            for row_idx, pattern in enumerate(obs_patterns):
                for col_idx, (obs_method, obs_ratio) in enumerate(col_configs):
                    ax = axes_agg[row_idx, col_idx]
                    
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
                    
                    # Prepare data for boxplot (aggregated)
                    data_for_plot = []
                    labels_for_plot = []
                    colors_for_plot = []
                    
                    for agg_method in agg_methods:
                        df_method = df_subplot[df_subplot['agg_method'] == agg_method]
                        if len(df_method) > 0:
                            data_for_plot.append(df_method['test_rmse'].values)
                            labels_for_plot.append(agg_method)
                            colors_for_plot.append(agg_colors[agg_method])
                    
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
            
            # Save aggregated figure
            agg_filename = data_file.replace('/', '_').replace('.csv', f'_agg_no_{param_to_ignore}.png')
            save_path_agg = output_dir / agg_filename
            plt.savefig(save_path_agg, dpi=300, bbox_inches='tight')
            print(f"     ✓ Aggregated plot (no {param_to_ignore}): {save_path_agg}")
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
