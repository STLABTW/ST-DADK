"""
Analyze Table 4.4 results

Summarizes CRPS results from Table 4.4 experiments and creates a table
matching the format in the thesis.
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
    
    # Try to load summary file first
    summary_path = results_dir / 'table_4_4_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            results = summary.get('results', [])
    else:
        # Load from individual experiment directories
        for scenario_dir in results_dir.iterdir():
            if not scenario_dir.is_dir():
                continue
            
            # Parse scenario and model from directory name
            parts = scenario_dir.name.split('_')
            if len(parts) >= 2:
                scenario_name = parts[0]
                model_name = '_'.join(parts[1:])
            else:
                continue
            
            # Load scenario summary if available
            scenario_summary_path = scenario_dir / 'scenario_summary.json'
            if scenario_summary_path.exists():
                with open(scenario_summary_path, 'r') as f:
                    scenario_data = json.load(f)
                    results.extend(scenario_data.get('results', []))
            else:
                # Load individual experiment results
                for exp_dir in scenario_dir.iterdir():
                    if not exp_dir.is_dir() or not exp_dir.name.startswith('exp_'):
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
    """
    Create Table 4.4 format from results
    
    Returns:
        pandas DataFrame matching Table 4.4 format
    """
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Group by scenario and model, compute mean and std of test_CRPS
    summary = []
    for scenario in ['Fixed_Uniform', 'Fixed_Clustered', 'Random_Uniform', 'Random_Clustered']:
        for model in ['STDK', 'DA-STDK']:
            mask = (df['scenario'] == scenario) & (df['model'] == model)
            subset = df[mask]
            
            if len(subset) > 0:
                crps_values = subset['test_crps'].values
                mean_crps = np.mean(crps_values)
                std_crps = np.std(crps_values)
                
                summary.append({
                    'Observation Scenario': scenario.replace('_', ' '),
                    'Observation Distribution': scenario.split('_')[1],
                    'Model': model,
                    'Mean CRPS': mean_crps,
                    'Std CRPS': std_crps,
                    'N': len(subset)
                })
    
    summary_df = pd.DataFrame(summary)
    
    # Pivot to match Table 4.4 format
    pivot_df = summary_df.pivot_table(
        index=['Observation Scenario', 'Observation Distribution'],
        columns='Model',
        values='Mean CRPS',
        aggfunc='first'
    )
    
    # Add std as additional rows or in parentheses
    pivot_std = summary_df.pivot_table(
        index=['Observation Scenario', 'Observation Distribution'],
        columns='Model',
        values='Std CRPS',
        aggfunc='first'
    )
    
    return pivot_df, pivot_std, summary_df


def print_table_4_4(pivot_df, pivot_std, summary_df):
    """
    Print Table 4.4 in a readable format
    """
    print("\n" + "="*80)
    print("TABLE 4.4: CRPS Comparison between STDK and DA-STDK")
    print("(Multi-quantile joint training, 2b-8 dataset)")
    print("="*80)
    print()
    
    # Print summary table
    print("Summary Statistics:")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    print()
    
    # Print pivot table
    print("Table 4.4 Format (Mean CRPS):")
    print("-" * 80)
    print(pivot_df.to_string())
    print()
    
    # Print with standard deviations
    print("Table 4.4 Format (Mean ± Std CRPS):")
    print("-" * 80)
    for idx in pivot_df.index:
        scenario, distribution = idx
        stddk_mean = pivot_df.loc[idx, 'STDK']
        stddk_std = pivot_std.loc[idx, 'STDK']
        dastdk_mean = pivot_df.loc[idx, 'DA-STDK']
        dastdk_std = pivot_std.loc[idx, 'DA-STDK']
        
        print(f"{scenario:20s} {distribution:15s}  "
              f"STDK: {stddk_mean:.6f} ± {stddk_std:.6f}  "
              f"DA-STDK: {dastdk_mean:.6f} ± {dastdk_std:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Table 4.4 results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Results directory from run_table_4_4.py'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Output CSV file path (optional)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return
    
    # Load results
    print(f"Loading results from: {results_dir}")
    results = load_table_4_4_results(results_dir)
    
    if len(results) == 0:
        print("Error: No results found")
        return
    
    print(f"Loaded {len(results)} experiment results")
    
    # Create table
    pivot_df, pivot_std, summary_df = create_table_4_4(results)
    
    # Print table
    print_table_4_4(pivot_df, pivot_std, summary_df)
    
    # Save to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to: {output_path}")
        
        # Also save pivot table
        pivot_path = output_path.parent / f"{output_path.stem}_pivot.csv"
        pivot_df.to_csv(pivot_path)
        print(f"Pivot table saved to: {pivot_path}")


if __name__ == '__main__':
    main()
