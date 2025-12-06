"""
Grid Search Experiment Runner

모든 하이퍼파라미터 조합에 대해 실험을 수행하고 결과를 CSV로 저장합니다.
"""
import argparse
import yaml
import itertools
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train_st_interp import run_single_experiment
import torch


def generate_config_combinations(base_config, param_grid, filter_fn=None):
    """
    Generate all combinations of parameters
    
    Args:
        base_config: Base configuration dictionary
        param_grid: Dictionary of parameters to vary
                   e.g., {'obs_method': ['site-wise', 'random'], 
                          'obs_ratio': [0.1, 0.3]}
        filter_fn: Optional function to filter combinations
                   Takes a dict of {param_name: param_value} and returns True to keep
    
    Returns:
        List of configuration dictionaries
    """
    # Get parameter names and values
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    
    # Create config for each combination
    configs = []
    config_counter = 0  # Counter for valid configs only
    
    for original_idx, combo in enumerate(combinations, 1):
        # Create param dict for filtering
        param_dict = dict(zip(param_names, combo))
        
        # Apply filter if provided
        if filter_fn is not None and not filter_fn(param_dict):
            continue
        
        config_counter += 1  # Increment only for kept configs
        
        config = base_config.copy()
        
        # Update with combination values
        for param_name, param_value in zip(param_names, combo):
            config[param_name] = param_value
        
        # Generate tag using config_counter instead of original_idx
        tag_parts = [f"config{config_counter:03d}"]
        for param_name, param_value in zip(param_names, combo):
            # Shorten parameter values for tag
            if param_name == 'spatial_init_method':
                tag_parts.append('uni' if param_value == 'uniform' else 'gmm')
            elif param_name == 'spatial_learnable':
                tag_parts.append('lrn' if param_value else 'fix')
            elif param_name == 'obs_method':
                tag_parts.append('site' if param_value == 'site-wise' else 'rand')
            elif param_name == 'obs_ratio':
                tag_parts.append(f'{int(param_value*100)}')
            elif param_name == 'obs_spatial_pattern':
                tag_parts.append('cor' if param_value == 'corner' else 'unf')
            else:
                tag_parts.append(str(param_value))
        
        config['tag'] = '_'.join(tag_parts)
        config['config_id'] = config_counter  # Use config_counter
        
        configs.append(config)
    
    return configs


def save_experiment_results(all_results, output_dir):
    """
    Save experiment results to CSV files
    
    Args:
        all_results: List of result dictionaries from each config
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary statistics per config
    summary_records = []
    
    for result in all_results:
        if result is None:
            continue
            
        summary = result['summary']
        config = result['config']
        
        record = {
            'config_id': config['config_id'],
            'tag': config['tag'],
            'spatial_init_method': config['spatial_init_method'],
            'spatial_learnable': config['spatial_learnable'],
            'obs_method': config['obs_method'],
            'obs_ratio': config['obs_ratio'],
            'obs_spatial_pattern': config['obs_spatial_pattern'],
            'n_experiments': summary['n_experiments'],
        }
        
        # Add statistics
        for metric in ['test_rmse', 'test_mae', 'test_mse', 
                       'valid_rmse', 'valid_mae', 'valid_mse',
                       'train_rmse', 'train_mae', 'train_mse',
                       'total_time_seconds']:
            if metric in summary['statistics']:
                stats = summary['statistics'][metric]
                record[f'{metric}_mean'] = stats['mean']
                record[f'{metric}_std'] = stats['std']
                record[f'{metric}_min'] = stats['min']
                record[f'{metric}_max'] = stats['max']
                record[f'{metric}_median'] = stats['median']
        
        summary_records.append(record)
    
    # Save summary
    df_summary = pd.DataFrame(summary_records)
    summary_file = output_dir / 'grid_search_summary.csv'
    df_summary.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved: {summary_file}")
    
    # 2. Detailed results per iteration
    detail_records = []
    
    for result in all_results:
        if result is None:
            continue
            
        summary = result['summary']
        config = result['config']
        
        # Extract individual experiment values
        for metric in ['test_rmse', 'test_mae', 'test_mse',
                       'valid_rmse', 'valid_mae', 'valid_mse',
                       'train_rmse', 'train_mae', 'train_mse',
                       'total_time_seconds']:
            if metric in summary['statistics']:
                values = summary['statistics'][metric]['values']
                
                for exp_id, value in enumerate(values, 1):
                    # Find or create record for this experiment
                    record = None
                    for r in detail_records:
                        if r['config_id'] == config['config_id'] and r['experiment_id'] == exp_id:
                            record = r
                            break
                    
                    if record is None:
                        record = {
                            'config_id': config['config_id'],
                            'tag': config['tag'],
                            'experiment_id': exp_id,
                            'spatial_init_method': config['spatial_init_method'],
                            'spatial_learnable': config['spatial_learnable'],
                            'obs_method': config['obs_method'],
                            'obs_ratio': config['obs_ratio'],
                            'obs_spatial_pattern': config['obs_spatial_pattern'],
                        }
                        detail_records.append(record)
                    
                    record[metric] = value
    
    # Save detailed results
    df_detail = pd.DataFrame(detail_records)
    detail_file = output_dir / 'grid_search_detail.csv'
    df_detail.to_csv(detail_file, index=False)
    print(f"✓ Detailed results saved: {detail_file}")
    
    # 3. Save configuration information
    config_records = []
    configs_dict = {}
    for result in all_results:
        if result is None:
            continue
        config = result['config']
        config_records.append({
            'config_id': config['config_id'],
            'tag': config['tag'],
        })
        configs_dict[str(config['config_id'])] = config
    
    # Save configs as separate JSON file (cleaner than CSV with escaped quotes)
    config_json_file = output_dir / 'grid_search_configs.json'
    with open(config_json_file, 'w', encoding='utf-8') as f:
        json.dump(configs_dict, f, indent=2, ensure_ascii=False)
    print(f"✓ Configurations saved: {config_json_file}")
    
    # Also save a simple CSV with just IDs and tags
    df_configs = pd.DataFrame(config_records)
    config_file = output_dir / 'grid_search_configs.csv'
    df_configs.to_csv(config_file, index=False)
    print(f"✓ Configuration index saved: {config_file}")
    
    return df_summary, df_detail


def main():
    parser = argparse.ArgumentParser(description='Grid Search Experiment Runner')
    parser.add_argument('--config', type=str, default='configs/config_st_interp.yaml',
                       help='Base configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: results/<date>_grid_search)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel')
    parser.add_argument('--n_jobs', type=int, default=10,
                       help='Number of parallel jobs')
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # Define parameter grid
    param_grid = {
        'data_file': ['data/2b/2b_8.csv'],
        'spatial_init_method': ['uniform', 'gmm'],
        'spatial_learnable': [True, False],
        'obs_method': ['site-wise', 'random'],
        'obs_ratio': [0.05, 0.25],
        'obs_spatial_pattern': ['corner', 'uniform'],
    }
    
    # Define filter function for conditional combinations
    # Example: uniform -> fixed, gmm -> learnable
    def config_filter(params):
        """Filter configurations based on conditions"""
        # If you want only: uniform+fixed and gmm+learnable
        if params['spatial_init_method'] == 'uniform' and params['spatial_learnable'] == True:
            return False
        if params['spatial_init_method'] == 'gmm' and params['spatial_learnable'] == False:
            return False
        return True
    
    # Generate all config combinations
    print("\n" + "="*100)
    print("GRID SEARCH EXPERIMENT RUNNER")
    print("="*100)
    print(f"\nBase config: {args.config}")
    print(f"\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"\nFilter: uniform+fixed and gmm+learnable only")
    
    configs = generate_config_combinations(base_config, param_grid, filter_fn=config_filter)
    n_configs = len(configs)
    n_experiments_per_config = base_config.get('n_experiments', 10)
    total_experiments = n_configs * n_experiments_per_config
    
    print(f"\nTotal configurations: {n_configs}")
    print(f"Experiments per config: {n_experiments_per_config}")
    print(f"Total experiments: {total_experiments}")
    
    # Setup output directory
    if args.output_dir is None:
        datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/{datetime_str}_grid_search'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Parallel execution: {args.parallel}")
    if args.parallel:
        print(f"Number of parallel jobs: {args.n_jobs}")
    
    # Device
    device = base_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n" + "="*100)
    
    # Run experiments
    all_results = []
    
    if args.parallel:
        # Parallel execution
        from joblib import Parallel, delayed
        
        def run_config_wrapper(config, config_idx):
            """Wrapper for parallel execution"""
            import warnings
            warnings.filterwarnings('ignore')
            
            print(f"\n[{config_idx}/{n_configs}] Running {config['tag']}...")
            
            # Create output directory for this config
            config_output_dir = output_dir / config['tag']
            config_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config
            with open(config_output_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Run experiments for this config
            try:
                from train_st_interp import run_multiple_experiments
                summary = run_multiple_experiments(
                    config, 
                    config_output_dir, 
                    device,
                    parallel=True
                )
                
                return {
                    'config': config,
                    'summary': summary,
                    'status': 'success'
                }
            except Exception as e:
                print(f"✗ Config {config['tag']} FAILED: {e}")
                return {
                    'config': config,
                    'summary': None,
                    'status': 'failed',
                    'error': str(e)
                }
        
        print(f"\nRunning {n_configs} configs in parallel...")
        results_list = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(run_config_wrapper)(config, i) 
            for i, config in enumerate(configs, 1)
        )
        all_results = results_list
        
    else:
        # Sequential execution
        for i, config in enumerate(configs, 1):
            print(f"\n" + "="*100)
            print(f"[{i}/{n_configs}] Running Config: {config['tag']}")
            print("="*100)
            
            # Create output directory for this config
            config_output_dir = output_dir / config['tag']
            config_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config
            with open(config_output_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Run experiments for this config
            try:
                from train_st_interp import run_multiple_experiments
                summary = run_multiple_experiments(
                    config, 
                    config_output_dir, 
                    device,
                    parallel=False
                )
                
                all_results.append({
                    'config': config,
                    'summary': summary,
                    'status': 'success'
                })
                
                print(f"\n✓ Config {i}/{n_configs} completed")
                
            except Exception as e:
                print(f"\n✗ Config {i}/{n_configs} FAILED: {e}")
                import traceback
                traceback.print_exc()
                
                all_results.append({
                    'config': config,
                    'summary': None,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Save results
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)
    
    df_summary, df_detail = save_experiment_results(all_results, output_dir)
    
    # Print summary
    n_success = sum(1 for r in all_results if r['status'] == 'success')
    n_failed = len(all_results) - n_success
    
    print("\n" + "="*100)
    print("GRID SEARCH COMPLETE!")
    print("="*100)
    print(f"Total configs: {n_configs}")
    print(f"Successful: {n_success}")
    print(f"Failed: {n_failed}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - grid_search_summary.csv: Summary statistics per config")
    print(f"  - grid_search_detail.csv: Detailed results per iteration")
    print(f"  - grid_search_configs.csv: Configuration details")
    print("="*100)


if __name__ == '__main__':
    main()
