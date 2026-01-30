#!/usr/bin/env python3
"""
Run Table 4.4: all scenarios × models with N replicates each.

Scenarios: Fixed_Uniform, Fixed_Clustered, Random_Uniform, Random_Clustered
Models: STDK, DA-STDK
Output: results/table_4_4_<timestamp>/<Scenario>_<Model>/ (each with experiments/1..N)

Forces: split_method=random, calibration_ratio_from_train=0.2, calibration_split_method=random.
With train_ratio=0.8, calibration uses validation split.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Scenario: (obs_method, obs_spatial_pattern)
SCENARIOS = {
    'Fixed_Uniform':     ('site-wise', 'uniform'),
    'Fixed_Clustered':   ('site-wise', 'corner'),
    'Random_Uniform':    ('random', 'uniform'),
    'Random_Clustered':  ('random', 'corner'),
}
# Model: (spatial_learnable, spatial_init_method)
MODELS = {
    'STDK':     (False, 'uniform'),
    'DA-STDK':  (True, 'gmm'),
}


def main():
    ap = argparse.ArgumentParser(description='Run Table 4.4 (all scenarios × models, N replicates).')
    ap.add_argument('--config', type=str, default='configs/config_st_interp.yaml', help='Base config YAML')
    ap.add_argument('--n_experiments', type=int, default=10, help='Replicates per scenario×model')
    ap.add_argument('--data_file', type=str, default='data/2b/2b_8.csv', help='Data file (2b for Table 4.4)')
    ap.add_argument('--base_seed', type=int, default=2025)
    ap.add_argument('--parallel', action='store_true', help='Run replicates in parallel per scenario×model')
    ap.add_argument('--dry-run', action='store_true', help='Print commands only')
    ap.add_argument('--analyze', action='store_true', help='Run analyze_table_4_4.py after all runs finish')
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / 'scripts' / 'train_st_interp.py'
    analyze_script = repo_root / 'scripts' / 'analyze_table_4_4.py'
    if not train_script.exists():
        train_script = repo_root / 'train_st_interp.py'
    if not train_script.exists():
        print(f'Not found: {train_script}', file=sys.stderr)
        sys.exit(1)

    base_config_path = repo_root / args.config
    if not base_config_path.exists():
        print(f'Config not found: {base_config_path}', file=sys.stderr)
        sys.exit(1)

    import yaml
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    # Table 4.4 overrides (coverage setup)
    base_config['data_file'] = args.data_file
    base_config['split_method'] = 'random'
    base_config['train_ratio'] = 0.8
    base_config['calibration_ratio_from_train'] = 0.2
    base_config['calibration_split_method'] = 'random'
    base_config['n_experiments'] = args.n_experiments
    base_config['base_seed'] = args.base_seed

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_base = repo_root / 'results' / f'table_4_4_{timestamp}'
    results_base.mkdir(parents=True, exist_ok=True)
    print(f'Results base: {results_base}')

    for scenario_name, (obs_method, obs_spatial_pattern) in SCENARIOS.items():
        for model_name, (spatial_learnable, spatial_init_method) in MODELS.items():
            combo_name = f'{scenario_name}_{model_name}'
            config = dict(base_config)
            config['obs_method'] = obs_method
            config['obs_spatial_pattern'] = obs_spatial_pattern
            config['spatial_learnable'] = spatial_learnable
            config['spatial_init_method'] = spatial_init_method
            config['tag'] = f'table4.4_{combo_name}'

            out_dir = results_base / combo_name
            out_dir.mkdir(parents=True, exist_ok=True)
            config_path = out_dir / 'config.yaml'
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)

            cmd = [
                sys.executable,
                str(train_script),
                '--config', str(config_path),
                '--output_dir', str(out_dir),
                '--n_experiments', str(args.n_experiments),
                '--base_seed', str(args.base_seed),
            ]
            if args.parallel:
                cmd.append('--parallel')
            print(f'Run: {combo_name}')
            if args.dry_run:
                print(' ', ' '.join(cmd))
                continue
            env = os.environ.copy()
            env['PYTHONPATH'] = str(repo_root)
            ret = subprocess.run(cmd, cwd=str(repo_root), env=env)
            if ret.returncode != 0:
                print(f'Failed: {combo_name} (exit {ret.returncode})', file=sys.stderr)
                sys.exit(ret.returncode)

    print(f'Done. Results: {results_base}')

    if args.analyze and not args.dry_run and analyze_script.exists():
        print('Running analyzer...')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(repo_root)
        ret = subprocess.run(
            [sys.executable, str(analyze_script), '--results_dir', str(results_base)],
            cwd=str(repo_root), env=env
        )
        if ret.returncode != 0:
            print(f'Analyzer exited with {ret.returncode}', file=sys.stderr)
    else:
        print(f'Analyze: PYTHONPATH=. python scripts/analyze_table_4_4.py --results_dir {results_base}')


if __name__ == '__main__':
    main()
