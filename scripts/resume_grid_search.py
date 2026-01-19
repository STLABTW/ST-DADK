#!/usr/bin/env python
"""Resume grid search for specific experiments (e.g., 17-20)"""

import argparse
import yaml
import json
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from train_st_interp import run_multiple_experiments
import torch


def main():
    parser = argparse.ArgumentParser(description='Resume grid search for specific experiments')
    parser.add_argument('--grid_dir', type=str, required=True, 
                      help='Grid search directory (e.g., 20251206_224336_grid_search)')
    parser.add_argument('--start_exp_id', type=int, default=None,
                      help='Starting experiment ID (e.g., 17). Not required if --summarize-only')
    parser.add_argument('--end_exp_id', type=int, default=None,
                      help='Ending experiment ID (e.g., 20). Not required if --summarize-only')
    parser.add_argument('--config_filter', type=str, default=None,
                      help='Optional: only process configs matching this pattern (e.g., "config001")')
    parser.add_argument('--summarize-only', action='store_true',
                      help='Only regenerate summaries from existing results, do not run experiments')
    parser.add_argument('--skip-existing', action='store_true',
                      help='Skip experiments that already have results.json (only run missing ones)')
    args = parser.parse_args()
    
    grid_dir = Path(args.grid_dir)
    if not grid_dir.exists():
        print(f"[ERROR] Grid directory not found: {grid_dir}")
        return
    
    # Validate arguments
    if not args.summarize_only:
        if args.start_exp_id is None or args.end_exp_id is None:
            print(f"[ERROR] --start_exp_id and --end_exp_id are required unless --summarize-only is used")
            return
    
    # Find all actual config directories (with config.yaml inside)
    # These may be nested like config001_data/2b/2b_8.csv_uni_fix_site_5_cor/
    import glob
    config_files = sorted(grid_dir.glob('**/config.yaml'))
    config_dirs = [f.parent for f in config_files if 'experiments' in [d.name for d in f.parent.iterdir()]]
    
    if args.config_filter:
        config_dirs = [d for d in config_dirs if args.config_filter in str(d)]
    
    print(f"\n{'='*70}")
    if args.summarize_only:
        print(f"SUMMARIZE GRID SEARCH RESULTS")
    else:
        print(f"RESUME GRID SEARCH")
    print(f"{'='*70}")
    print(f"Grid directory: {grid_dir}")
    if not args.summarize_only:
        print(f"Experiment range: {args.start_exp_id} to {args.end_exp_id}")
    else:
        print(f"Mode: Summary only (no experiments will be run)")
    print(f"Config directories to process: {len(config_dirs)}")
    if args.config_filter:
        print(f"Config filter: {args.config_filter}")
    print(f"{'='*70}\n")
    
    if args.summarize_only:
        # Just regenerate summaries from existing experiments
        print(f"Regenerating summaries from existing experiments...\n")
        
        for i, config_dir in enumerate(config_dirs, 1):
            rel_path = config_dir.relative_to(grid_dir)
            print(f"[{i}/{len(config_dirs)}] Processing {rel_path}...")
            
            # Load config
            config_file = config_dir / 'config.yaml'
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Regenerate summary from existing results
            try:
                summary = regenerate_summary_from_existing(config_dir, config)
                if summary:
                    n_exp = summary.get('n_experiments', 0)
                    test_rmse_mean = summary.get('statistics', {}).get('test_rmse', {}).get('mean', 0)
                    test_rmse_std = summary.get('statistics', {}).get('test_rmse', {}).get('std', 0)
                    print(f"  [OK] Summary regenerated: {n_exp} experiments")
                    print(f"    RMSE: {test_rmse_mean:.4f} ± {test_rmse_std:.4f}")
                else:
                    print(f"  [WARNING] No completed experiments found")
            except Exception as e:
                print(f"  [FAILED] FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    else:
        # Run experiments
        device = 'cpu'  # Force CPU to avoid CUDA errors
        print(f"Using device: {device} (forced)\n")
        
        for i, config_dir in enumerate(config_dirs, 1):
            # Display relative path for clarity
            rel_path = config_dir.relative_to(grid_dir)
            print(f"\n[{i}/{len(config_dirs)}] Processing {rel_path}...")
            
            # Load config
            config_file = config_dir / 'config.yaml'
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Run experiments for this config
            try:
                print(f"  Running experiments {args.start_exp_id}-{args.end_exp_id}...")
                print(f"  Config n_experiments: {config.get('n_experiments')}")
                print(f"  Config parallel: {config.get('parallel')}")
                print(f"  Config n_jobs: {config.get('n_jobs')}")
                
                summary = run_multiple_experiments(
                    config=config,
                    output_dir=config_dir,
                    device=device,
                    parallel=True,
                    start_exp_id=args.start_exp_id,
                    end_exp_id=args.end_exp_id,
                    skip_existing=args.skip_existing
                )
                
                if summary:
                    n_exp = summary.get('n_experiments', 0)
                    test_rmse_mean = summary.get('statistics', {}).get('test_rmse', {}).get('mean', 0)
                    test_rmse_std = summary.get('statistics', {}).get('test_rmse', {}).get('std', 0)
                    print(f"  [OK] Completed successfully: {n_exp} experiments")
                    print(f"    RMSE: {test_rmse_mean:.4f} ± {test_rmse_std:.4f}")
                else:
                    print(f"  [WARNING] Completed but no summary generated")
                    
            except Exception as e:
                print(f"  [FAILED] FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*70}")
    print(f"RESUME COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll config directories have been processed.")
    print(f"Summary files have been updated in each config directory.")
    
    # Generate grid-level summary files
    print(f"\n{'='*70}")
    print(f"GENERATING GRID SEARCH SUMMARY")
    print(f"{'='*70}")
    
    try:
        generate_grid_summary(grid_dir, config_dirs)
        print(f"\n[OK] Grid search summary files generated successfully!")
        print(f"\nTo analyze results, run:")
        print(f"  python scripts/analyze_grid_search.py {grid_dir}")
    except Exception as e:
        print(f"\n[WARNING] Failed to generate grid summary: {e}")
        import traceback
        traceback.print_exc()


def regenerate_summary_from_existing(config_dir, config):
    """Regenerate summary from existing experiment results"""
    from train_st_interp import aggregate_results
    
    n_experiments = config.get('n_experiments', 20)
    experiments_dir = config_dir / 'experiments'
    
    if not experiments_dir.exists():
        return None
    
    # Load all existing experiment results
    all_results = []
    for i in range(1, n_experiments + 1):
        result_file = experiments_dir / str(i) / 'results.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
    
    if len(all_results) == 0:
        return None
    
    # Regenerate summary
    summary_dir = config_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary = aggregate_results(all_results, summary_dir)
    return summary


def generate_grid_summary(grid_dir, config_dirs):
    """Generate grid-level summary files from all config summaries"""
    
    all_summaries = []
    
    # Load all config summaries
    for config_dir in config_dirs:
        config_file = config_dir / 'config.yaml'
        summary_json = config_dir / 'summary' / 'summary_statistics.json'  # Load JSON instead of CSV
        
        if not summary_json.exists():
            print(f"  [WARNING] Skipping {config_dir.name}: no summary found")
            continue
        
        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load summary statistics from JSON (includes values)
        with open(summary_json, 'r') as f:
            summary = json.load(f)
        
        all_summaries.append({
            'config': config,
            'summary': summary
        })
    
    if len(all_summaries) == 0:
        print("[WARNING] No summaries found to aggregate")
        return
    
    print(f"Found {len(all_summaries)} config summaries to aggregate")
    
    # 1. Summary statistics per config
    summary_records = []
    
    for result in all_summaries:
        summary = result['summary']
        config = result['config']
        
        record = {
            'config_id': config.get('config_id', 'unknown'),
            'tag': config.get('tag', 'unknown'),
            'spatial_basis_function': config.get('spatial_basis_function', 'wendland'),  # Added
            'spatial_init_method': config.get('spatial_init_method'),
            'spatial_learnable': config.get('spatial_learnable'),
            'obs_method': config.get('obs_method'),
            'obs_ratio': config.get('obs_ratio'),
            'obs_spatial_pattern': config.get('obs_spatial_pattern'),
            'n_experiments': summary['n_experiments'],
        }
        
        # Add statistics
        for metric in ['test_rmse', 'test_mae', 'test_mse',
                       'valid_rmse', 'valid_mae', 'valid_mse',
                       'train_rmse', 'train_mae', 'train_mse',
                       'total_time_seconds',
                       'train_crps', 'test_crps', 'valid_crps',  # Quantile metrics
                       'train_check_loss', 'test_check_loss', 'valid_check_loss']:
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
    summary_file = grid_dir / 'grid_search_summary.csv'
    df_summary.to_csv(summary_file, index=False)
    print(f"[OK] Summary saved: {summary_file}")
    
    # 2. Detailed results per iteration
    detail_records = []
    
    print(f"\nProcessing detailed results...")
    
    for result in all_summaries:
        summary = result['summary']
        config = result['config']
        
        print(f"  Processing config {config.get('config_id')}: {config.get('tag')}")
        
        # Extract individual experiment values from statistics
        for metric in ['test_rmse', 'test_mae', 'test_mse',
                       'valid_rmse', 'valid_mae', 'valid_mse',
                       'train_rmse', 'train_mae', 'train_mse',
                       'total_time_seconds']:
            if metric in summary['statistics']:
                values = summary['statistics'][metric].get('values', [])
                print(f"    {metric}: {len(values)} values")
                
                for exp_id, value in enumerate(values, 1):
                    # Find or create record for this experiment
                    record = None
                    for r in detail_records:
                        if r['config_id'] == config.get('config_id') and r['experiment_id'] == exp_id:
                            record = r
                            break
                    
                    if record is None:
                        record = {
                            'config_id': config.get('config_id'),
                            'tag': config.get('tag'),
                            'experiment_id': exp_id,
                            'spatial_basis_function': config.get('spatial_basis_function', 'wendland'),  # Added
                            'spatial_init_method': config.get('spatial_init_method'),
                            'spatial_learnable': config.get('spatial_learnable'),
                            'obs_method': config.get('obs_method'),
                            'obs_ratio': config.get('obs_ratio'),
                            'obs_spatial_pattern': config.get('obs_spatial_pattern'),
                        }
                        detail_records.append(record)
                    
                    record[metric] = value
    
    # Save detailed results
    if detail_records:
        df_detail = pd.DataFrame(detail_records)
        detail_file = grid_dir / 'grid_search_detail.csv'
        df_detail.to_csv(detail_file, index=False)
        print(f"[OK] Detailed results saved: {detail_file}")
    
    # 3. Save configs as JSON
    configs_dict = {}
    config_records = []
    
    for result in all_summaries:
        config = result['config']
        config_id = str(config.get('config_id', 'unknown'))
        configs_dict[config_id] = config
        config_records.append({
            'config_id': config.get('config_id'),
            'tag': config.get('tag')
        })
    
    config_json_file = grid_dir / 'grid_search_configs.json'
    with open(config_json_file, 'w', encoding='utf-8') as f:
        json.dump(configs_dict, f, indent=2, ensure_ascii=False)
    print(f"[OK] Configurations saved: {config_json_file}")
    
    df_configs = pd.DataFrame(config_records)
    config_file = grid_dir / 'grid_search_configs.csv'
    df_configs.to_csv(config_file, index=False)
    print(f"[OK] Configuration index saved: {config_file}")


if __name__ == '__main__':
    main()
