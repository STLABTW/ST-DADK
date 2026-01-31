"""
Analyze Table 4.4 results

Summarizes CRPS results from Table 4.4 experiments and creates a table
matching the format in the thesis. Includes coverage (nominal/conformal, qhat).
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def load_table_4_4_results(results_dir: Path):
    """
    Load all results from Table 4.4 experiments

    Returns:
        List of result dictionaries
    """
    results = []

    # Try to load summary file first (check both possible filenames)
    summary_path = results_dir / 'table_4_4_summary.json'
    if not summary_path.exists():
        summary_path = results_dir / 'result.json'

    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            results = summary.get('results', [])
    else:
        # Load from individual scenario directories (e.g. Fixed_Uniform_STDK/)
        for scenario_dir in results_dir.iterdir():
            if not scenario_dir.is_dir():
                continue

            # Parse scenario and model from directory name
            name = scenario_dir.name
            if name.endswith('_DA-STDK'):
                scenario_name = name[:-8]
                model_name = 'DA-STDK'
            elif name.endswith('_STDK'):
                scenario_name = name[:-5]
                model_name = 'STDK'
            else:
                parts = name.split('_')
                if len(parts) >= 2:
                    scenario_name = parts[0]
                    model_name = '_'.join(parts[1:])
                else:
                    continue

            scenario_summary_path = scenario_dir / 'scenario_summary.json'
            if scenario_summary_path.exists():
                with open(scenario_summary_path, 'r') as f:
                    scenario_data = json.load(f)
                    results.extend(scenario_data.get('results', []))
            else:
                # train_st_interp layout: experiments/1/, experiments/2/, ...
                exp_parent = scenario_dir / 'experiments'
                if exp_parent.exists():
                    search_dirs = list(exp_parent.iterdir())
                else:
                    search_dirs = list(scenario_dir.iterdir())
                for exp_dir in search_dirs:
                    if not exp_dir.is_dir():
                        continue
                    results_path = exp_dir / 'results.json'
                    if results_path.exists():
                        with open(results_path, 'r') as f:
                            result = json.load(f)
                            result['scenario'] = scenario_name
                            result['model'] = model_name
                            results.append(result)

    return results


def create_table_4_4(results: list):
    """Build Table 4.4 summary and pivots (CRPS + coverage)."""
    df = pd.DataFrame(results)
    summary = []
    for scenario in ['Fixed_Uniform', 'Fixed_Clustered', 'Random_Uniform', 'Random_Clustered']:
        for model in ['STDK', 'DA-STDK']:
            mask = (df['scenario'] == scenario) & (df['model'] == model)
            subset = df[mask]

            if len(subset) > 0:
                crps_values = subset['test_crps'].values
                mean_crps = np.mean(crps_values)
                std_crps = np.std(crps_values)
                test_cov = subset.get('test_coverage_90')
                test_cov_conf = subset.get('test_coverage_90_conformal')
                test_cov_global = subset.get('test_coverage_90_conformal_global')
                test_cov_cluster = subset.get('test_coverage_90_conformal_cluster')
                qhat_vals = subset.get('conformal_qhat')
                mean_qhat_global = subset.get('mean_qhat_global')
                mean_qhat_cluster = subset.get('mean_qhat_cluster')
                # Prefer calibration_coverage_90 (works for cal-from-train when train_ratio=1.0); fallback to valid
                cal_cov = subset.get('calibration_coverage_90')
                if cal_cov is None:
                    cal_cov = subset.get('valid_coverage_90_conformal')

                def _mean_std(series):
                    if series is None:
                        return (np.nan, np.nan)
                    values = pd.to_numeric(series, errors='coerce').dropna().values
                    if len(values) == 0:
                        return (np.nan, np.nan)
                    return (np.mean(values), np.std(values))

                mean_test_cov, std_test_cov = _mean_std(test_cov)
                mean_test_cov_conf, std_test_cov_conf = _mean_std(test_cov_conf)
                mean_test_cov_global, _ = _mean_std(test_cov_global)
                mean_test_cov_cluster, _ = _mean_std(test_cov_cluster)
                mean_qhat, std_qhat = _mean_std(qhat_vals)
                mqg, _ = _mean_std(mean_qhat_global)
                mqc, _ = _mean_std(mean_qhat_cluster)
                mean_cal_cov, std_cal_cov = _mean_std(cal_cov)

                summary.append({
                    'Observation Scenario': scenario.replace('_', ' '),
                    'Observation Distribution': scenario.split('_')[1],
                    'Model': model,
                    'Mean CRPS': mean_crps,
                    'Std CRPS': std_crps,
                    'Mean Test Coverage 90': mean_test_cov,
                    'Std Test Coverage 90': std_test_cov,
                    'Mean Test Coverage 90 (Conformal)': mean_test_cov_conf,
                    'Std Test Coverage 90 (Conformal)': std_test_cov_conf,
                    'Mean Test Coverage 90 (Conformal Global)': mean_test_cov_global,
                    'Mean Test Coverage 90 (Conformal Cluster)': mean_test_cov_cluster,
                    'Mean Conformal Qhat': mean_qhat,
                    'Std Conformal Qhat': std_qhat,
                    'Mean Qhat Global': mqg,
                    'Mean Qhat Cluster': mqc,
                    'Mean Calibration Coverage 90': mean_cal_cov,
                    'Std Calibration Coverage 90': std_cal_cov,
                    'N': len(subset)
                })

    summary_df = pd.DataFrame(summary)
    pivot_df = summary_df.pivot_table(
        index=['Observation Scenario', 'Observation Distribution'],
        columns='Model',
        values='Mean CRPS',
        aggfunc='first'
    )
    pivot_std = summary_df.pivot_table(
        index=['Observation Scenario', 'Observation Distribution'],
        columns='Model',
        values='Std CRPS',
        aggfunc='first'
    )
    row_order = [
        ('Fixed', 'Uniform'), ('Fixed', 'Clustered'),
        ('Random', 'Uniform'), ('Random', 'Clustered'),
    ]
    idx = pd.MultiIndex.from_tuples(
        [(f"{s} {d}", d) for s, d in row_order],
        names=['Observation Scenario', 'Observation Distribution']
    )
    pivot_df = pivot_df.reindex(idx).dropna(how='all')
    pivot_std = pivot_std.reindex(idx).dropna(how='all')
    pivot_df = pivot_df.reindex(columns=['STDK', 'DA-STDK'])
    pivot_std = pivot_std.reindex(columns=['STDK', 'DA-STDK'])
    return pivot_df, pivot_std, summary_df


def print_table_4_4(pivot_df, pivot_std, summary_df):
    """Print Table 4.4 and coverage summary."""
    print("\n" + "="*80)
    print("TABLE 4.4: CRPS Comparison between STDK and DA-STDK")
    print("(Multi-quantile joint training, 2b-8 dataset)")
    print("="*80)
    print()
    print("Table 4.4 (paper-style, Mean ± Std CRPS):")
    print("-" * 80)
    print(f"{'Scenario':<24} {'STDK':>14} {'DA-STDK':>14}")
    print("-" * 80)
    for idx in pivot_df.index:
        scenario, distribution = idx
        design = scenario.replace(' ' + distribution, '')
        label = f"{design} / {distribution}"
        def _fmt(col):
            m, s = pivot_df.loc[idx, col], pivot_std.loc[idx, col]
            if pd.isna(m) or pd.isna(s):
                return "      —"
            return f"{m:.4f} ± {s:.4f}"
        print(f"{label:<24} {_fmt('STDK'):>14}  {_fmt('DA-STDK'):>14}")
    print()
    print("Summary Statistics (all scenarios × models):")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    print()
    print("Pivot (Mean CRPS only):")
    print("-" * 80)
    print(pivot_df.to_string())
    print()
    if 'Mean Test Coverage 90 (Conformal)' in summary_df.columns:
        print("Coverage Summary (Mean ± Std):")
        print("-" * 80)
        cols = [
            'Observation Scenario', 'Observation Distribution', 'Model', 'N',
            'Mean Test Coverage 90', 'Std Test Coverage 90',
            'Mean Test Coverage 90 (Conformal)', 'Std Test Coverage 90 (Conformal)',
            'Mean Test Coverage 90 (Conformal Global)', 'Mean Test Coverage 90 (Conformal Cluster)',
            'Mean Conformal Qhat', 'Std Conformal Qhat',
            'Mean Qhat Global', 'Mean Qhat Cluster',
            'Mean Calibration Coverage 90', 'Std Calibration Coverage 90'
        ]
        available = [c for c in cols if c in summary_df.columns]
        print(summary_df[available].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Analyze Table 4.4 results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Results directory from run_table_4_4.py')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV file path (optional)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    print(f"Loading results from: {results_dir}")
    results = load_table_4_4_results(results_dir)
    if len(results) == 0:
        print("Error: No results found")
        return

    print(f"Loaded {len(results)} experiment results")
    pivot_df, pivot_std, summary_df = create_table_4_4(results)
    print_table_4_4(pivot_df, pivot_std, summary_df)

    if args.output_csv:
        output_path = Path(args.output_csv)
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to: {output_path}")
        pivot_path = output_path.parent / f"{output_path.stem}_pivot.csv"
        pivot_df.to_csv(pivot_path)
        print(f"Pivot table saved to: {pivot_path}")

    # Report spatial visualizations (from run_table_4_4 per-scenario aggregation)
    for scenario_dir in results_dir.iterdir():
        if scenario_dir.is_dir():
            summary_dir = scenario_dir / 'summary'
            if (summary_dir / 'spatial_coverage_aggregated.png').exists():
                print(f"\nSpatial coverage map: {scenario_dir.name}/summary/spatial_coverage_aggregated.png")
                break


if __name__ == '__main__':
    main()
