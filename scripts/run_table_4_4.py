"""
Table 4.4 Runner: Reproduce CRPS comparison between STDK and DA-STDK

This script runs the 4 scenarios from Table 4.4 in the thesis:
- Observation method: fixed (site-wise) vs random
- Observation distribution: uniform vs clustered
- Models: STDK vs DA-STDK
- Dataset: 2b-8
- Multi-quantile joint training with τ ∈ {0.05, 0.25, 0.5, 0.75, 0.95}

Usage:
    python scripts/run_table_4_4.py --config configs/config_st_interp.yaml --n_experiments 10
"""
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import sys
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_st_interp import run_single_experiment


def create_table_4_4_configs(base_config_path: str):
    """
    Create configurations for Table 4.4 scenarios
    
    Returns:
        List of (scenario_name, model_name, config_dict) tuples
    """
    # Load base config
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # Override base config for Table 4.4
    base_config['data_file'] = 'data/2b/2b_8.csv'  # 2b-8 dataset
    base_config['regression_type'] = 'multi-quantile'
    base_config['quantile_levels'] = [0.05, 0.25, 0.5, 0.75, 0.95]
    base_config['obs_ratio'] = 0.1  # 10% observation ratio (typical for experiments)
    
    # Force thesis-specific non-crossing setup (Section 4.2.2)
    # Use δ reparameterization with P_nc(δ) penalty as per Equation 3.9
    base_config['use_delta_reparameterization'] = True
    base_config['non_crossing_lambda'] = 1.0  # P_nc(δ) penalty weight λ (Section 3.2, Eq. 3.9)
    
    # Define 4 scenarios
    scenarios = [
        {
            'name': 'Fixed_Uniform',
            'obs_method': 'site-wise',
            'obs_spatial_pattern': 'uniform'
        },
        {
            'name': 'Fixed_Clustered',
            'obs_method': 'site-wise',
            'obs_spatial_pattern': 'corner'
        },
        {
            'name': 'Random_Uniform',
            'obs_method': 'random',
            'obs_spatial_pattern': 'uniform'
        },
        {
            'name': 'Random_Clustered',
            'obs_method': 'random',
            'obs_spatial_pattern': 'corner'
        }
    ]
    
    # Check if k_means_constrained is available for DA-STDK initialization
    try:
        import k_means_constrained
        da_stdk_init_method = 'kmeans_balanced'
    except ImportError:
        print("Warning: k_means_constrained not available. Using 'gmm' for DA-STDK initialization.")
        da_stdk_init_method = 'gmm'
    
    # Define 2 models
    models = [
        {
            'name': 'STDK',
            'spatial_init_method': 'uniform',
            'spatial_learnable': False
        },
        {
            'name': 'DA-STDK',
            'spatial_init_method': da_stdk_init_method,  # 'kmeans_balanced' if available, else 'gmm'
            'spatial_learnable': True
        }
    ]
    
    # Generate all combinations
    configs = []
    for scenario in scenarios:
        for model in models:
            config = base_config.copy()
            config['obs_method'] = scenario['obs_method']
            config['obs_spatial_pattern'] = scenario['obs_spatial_pattern']
            config['spatial_init_method'] = model['spatial_init_method']
            config['spatial_learnable'] = model['spatial_learnable']
            
            # Create experiment tag
            config['tag'] = f"table4.4_{scenario['name']}_{model['name']}"
            
            scenario_name = scenario['name']
            model_name = model['name']
            configs.append((scenario_name, model_name, config))
    
    return configs


def run_table_4_4_experiments(
    base_config_path: str,
    output_dir: str = None,
    n_experiments: int = 10,
    device: str = None,
    verbose: bool = True,
    parallel_mode: bool = False,
    skip_existing: bool = False
):
    """
    Run all Table 4.4 experiments
    
    Args:
        base_config_path: Path to base config file
        output_dir: Output directory (default: results/table_4_4_<timestamp>)
        n_experiments: Number of repeated experiments per scenario (default: 10)
        device: torch device (default: 'cuda' if available, else 'cpu')
        verbose: Whether to print detailed logs
        parallel_mode: Whether running in parallel mode
        skip_existing: If True, skip experiments that already have results
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"results/table_4_4_{timestamp}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate configurations
    configs = create_table_4_4_configs(base_config_path)
    
    print("="*80)
    print("TABLE 4.4 RUNNER: CRPS Comparison between STDK and DA-STDK")
    print("="*80)
    print(f"Dataset: 2b-8")
    print(f"Scenarios: 4 (Fixed/Random × Uniform/Clustered)")
    print(f"Models: 2 (STDK, DA-STDK)")
    print(f"Total experiments: {len(configs)} scenarios × {n_experiments} runs = {len(configs) * n_experiments}")
    print(f"Output directory: {output_path}")
    print(f"Device: {device}")
    print("="*80)
    
    # Save configuration summary
    summary = {
        'dataset': '2b-8',
        'scenarios': [c[0] for c in configs],
        'models': [c[1] for c in configs],
        'n_experiments': n_experiments,
        'quantile_levels': [0.05, 0.25, 0.5, 0.75, 0.95],
        'configs': []
    }
    
    # Run experiments
    all_results = []
    experiment_counter = 0
    
    for scenario_name, model_name, config in configs:
        # Update n_experiments in config
        config['n_experiments'] = n_experiments
        
        # Create scenario output directory
        scenario_output_dir = output_path / f"{scenario_name}_{model_name}"
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name} | Model: {model_name}")
        print(f"{'='*80}")
        print(f"  Observation method: {config['obs_method']}")
        print(f"  Observation pattern: {config['obs_spatial_pattern']}")
        print(f"  Spatial init: {config['spatial_init_method']}")
        print(f"  Spatial learnable: {config['spatial_learnable']}")
        print(f"  Output directory: {scenario_output_dir}")
        print(f"  Running {n_experiments} experiments...")
        
        # Save config for this scenario
        config_path = scenario_output_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        summary['configs'].append({
            'scenario': scenario_name,
            'model': model_name,
            'output_dir': str(scenario_output_dir),
            'config_path': str(config_path)
        })
        
        # Run experiments
        scenario_results = []
        for exp_id in range(1, n_experiments + 1):
            experiment_counter += 1
            exp_output_dir = scenario_output_dir / f"exp_{exp_id:03d}"
            
            if skip_existing and (exp_output_dir / 'results.json').exists():
                if verbose:
                    print(f"  Experiment {exp_id}/{n_experiments}: Skipping (already exists)")
                continue
            
            if verbose:
                print(f"\n  Experiment {exp_id}/{n_experiments} ({experiment_counter}/{len(configs) * n_experiments})")
            
            try:
                result = run_single_experiment(
                    config=config,
                    experiment_id=exp_id,
                    output_dir=exp_output_dir,
                    device=device,
                    verbose=verbose,
                    parallel_mode=parallel_mode,
                    skip_existing=skip_existing
                )
                
                # Add scenario and model info to result
                result['scenario'] = scenario_name
                result['model'] = model_name
                scenario_results.append(result)
                all_results.append(result)
                
                if verbose:
                    print(f"    ✓ Completed: test_CRPS = {result.get('test_crps', 'N/A'):.6f}")
            
            except Exception as e:
                print(f"    ✗ Failed: {str(e)}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Save scenario summary
        if scenario_results:
            scenario_summary_path = scenario_output_dir / 'scenario_summary.json'
            import json
            with open(scenario_summary_path, 'w') as f:
                json.dump({
                    'scenario': scenario_name,
                    'model': model_name,
                    'n_experiments': len(scenario_results),
                    'results': scenario_results
                }, f, indent=2, default=str)
    
    # Save overall summary
    summary_path = output_path / 'table_4_4_summary.json'
    import json
    summary['results'] = all_results
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("TABLE 4.4 EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nTo analyze results, use:")
    print(f"  python scripts/analyze_table_4_4.py --results_dir {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Table 4.4 Runner: Reproduce CRPS comparison between STDK and DA-STDK'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_st_interp.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: results/table_4_4_<timestamp>)'
    )
    parser.add_argument(
        '--n_experiments',
        type=int,
        default=10,
        help='Number of repeated experiments per scenario (default: 10)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce verbosity'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run experiments in parallel mode'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip experiments that already have results'
    )
    
    args = parser.parse_args()
    
    run_table_4_4_experiments(
        base_config_path=args.config,
        output_dir=args.output_dir,
        n_experiments=args.n_experiments,
        device=args.device,
        verbose=not args.quiet,
        parallel_mode=args.parallel,
        skip_existing=args.skip_existing
    )


if __name__ == '__main__':
    main()
