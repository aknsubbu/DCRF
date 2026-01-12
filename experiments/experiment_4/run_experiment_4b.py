"""
Experiment 4B: Sensitivity Analysis on Threshold

This script tests the 75% selection rule across different thresholds
(60%, 70%, 75%, 80%, 90%) to evaluate if 75% is optimal.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from 4A
from run_experiment_4a import (
    load_quadrant_data,
    load_variable_impacts,
    select_equal_count_for_target
)

# Set style
sns.set_style("whitegrid")


def run_sensitivity_analysis(method: str = 'avg_impact',
                             thresholds: List[float] = None):
    """
    Run sensitivity analysis across different impact thresholds.
    
    Parameters
    ----------
    method : str
        Method from Experiment 1/2
    thresholds : list
        List of threshold percentages (default: [0.6, 0.7, 0.75, 0.8, 0.9])
    """
    print("=" * 80)
    print("EXPERIMENT 4B: SENSITIVITY ANALYSIS ON THRESHOLD")
    print("=" * 80)
    print()
    
    if thresholds is None:
        thresholds = [0.60, 0.70, 0.75, 0.80, 0.90]
    
    # Load data
    exp2_results = Path(__file__).parent.parent / 'experiment_2' / 'results'
    exp1_results = Path(__file__).parent.parent / 'experiment_1' / 'results' / 'experiment_1a_results.csv'
    
    quadrant_data, total_impact = load_quadrant_data(exp2_results, method)
    impact_dict = load_variable_impacts(exp1_results, method)
    
    print(f"Method: {method}")
    print(f"Testing thresholds: {[f'{t*100}%' for t in thresholds]}")
    print()
    
    # Run selection for each threshold
    results = []
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Threshold: {threshold * 100}%")
        print('='*60)
        
        target_impact = threshold * total_impact
        
        selection = select_equal_count_for_target(
            quadrant_data['quadrants'],
            impact_dict,
            target_impact,
            total_impact
        )
        
        results.append({
            'threshold_pct': threshold * 100,
            'threshold': threshold,
            'target_impact': target_impact,
            'count_per_quadrant': selection['count_per_quadrant'],
            'total_selected': selection['total_selected'],
            'cumulative_impact': selection['cumulative_impact'],
            'coverage_pct': selection['coverage_pct'],
            'target_achieved': selection['target_achieved'],
            'diff_from_target': abs(selection['cumulative_impact'] - target_impact)
        })
        
        print(f"\nResults:")
        print(f"  Variables per quadrant: {selection['count_per_quadrant']}")
        print(f"  Total selected: {selection['total_selected']}")
        print(f"  Coverage: {selection['coverage_pct']:.1f}%")
        print(f"  Target achieved: {'✓' if selection['target_achieved'] else '✗'}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    print(results_df.to_string(index=False))
    
    # Save results
    output_path = Path(__file__).parent / 'results'
    output_path.mkdir(exist_ok=True, parents=True)
    
    csv_path = output_path / f'sensitivity_analysis_{method}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results: {csv_path}")
    
    # Visualize
    visualize_sensitivity(results_df, method, output_path)
    
    # Analysis and recommendation
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATION")
    print("=" * 80)
    
    analyze_threshold_optimality(results_df)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 4B COMPLETED")
    print("=" * 80)
    
    return results_df


def visualize_sensitivity(results_df: pd.DataFrame, method: str, output_dir: Path):
    """
    Create visualizations for sensitivity analysis.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    method : str
        Method name
    output_dir : Path
        Output directory
    """
    output_path = output_dir / 'figures'
    output_path.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Variables selected vs threshold
    ax = axes[0, 0]
    ax.plot(results_df['threshold_pct'], results_df['total_selected'],
            marker='o', linewidth=2.5, markersize=10, color='#2c3e50')
    ax.set_xlabel('Target Threshold (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Variables Selected', fontsize=12, fontweight='bold')
    ax.set_title('Variables Selected vs Threshold', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for _, row in results_df.iterrows():
        ax.annotate(f'{int(row["total_selected"])}',
                   xy=(row['threshold_pct'], row['total_selected']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontweight='bold')
    
    # 2. Coverage achieved vs target
    ax = axes[0, 1]
    ax.plot(results_df['threshold_pct'], results_df['coverage_pct'],
            marker='s', linewidth=2.5, markersize=10, color='#27ae60', label='Achieved')
    ax.plot(results_df['threshold_pct'], results_df['threshold_pct'],
            linestyle='--', linewidth=2, color='red', alpha=0.7, label='Target')
    ax.set_xlabel('Target Threshold (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Achieved vs Target Coverage', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Count per quadrant
    ax = axes[1, 0]
    ax.plot(results_df['threshold_pct'], results_df['count_per_quadrant'],
            marker='^', linewidth=2.5, markersize=10, color='#e74c3c')
    ax.set_xlabel('Target Threshold (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variables per Quadrant (Q1, Q2, Q3)', fontsize=12, fontweight='bold')
    ax.set_title('Equal Count Selection', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for _, row in results_df.iterrows():
        ax.annotate(f'{int(row["count_per_quadrant"])}',
                   xy=(row['threshold_pct'], row['count_per_quadrant']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontweight='bold')
    
    # 4. Prediction accuracy (how close to target)
    ax = axes[1, 1]
    results_df['error'] = abs(results_df['coverage_pct'] - results_df['threshold_pct'])
    
    bars = ax.bar(results_df['threshold_pct'], results_df['error'],
                  width=4, color='#f39c12', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Target Threshold (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (% points)', fontsize=12, fontweight='bold')
    ax.set_title('Selection Precision (Error from Target)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, error in zip(bars, results_df['error']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle(f'Sensitivity Analysis: Impact Threshold - {method.upper()}',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    fig_path = output_path / f'sensitivity_analysis_{method}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {fig_path}")
    
    plt.show()


def analyze_threshold_optimality(results_df: pd.DataFrame):
    """
    Analyze results and provide recommendation on optimal threshold.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    """
    print("\n1. Variable Count Analysis:")
    print(f"   60% threshold: {results_df[results_df['threshold_pct']==60]['total_selected'].values[0]} variables")
    print(f"   70% threshold: {results_df[results_df['threshold_pct']==70]['total_selected'].values[0]} variables")
    print(f"   75% threshold: {results_df[results_df['threshold_pct']==75]['total_selected'].values[0]} variables")
    print(f"   80% threshold: {results_df[results_df['threshold_pct']==80]['total_selected'].values[0]} variables")
    print(f"   90% threshold: {results_df[results_df['threshold_pct']==90]['total_selected'].values[0]} variables")
    
    print("\n2. Coverage Accuracy:")
    for _, row in results_df.iterrows():
        error = abs(row['coverage_pct'] - row['threshold_pct'])
        print(f"   {row['threshold_pct']:.0f}% target: Achieved {row['coverage_pct']:.1f}% (error: {error:.1f}%)")
    
    # Find most accurate
    results_df['error'] = abs(results_df['coverage_pct'] - results_df['threshold_pct'])
    best_idx = results_df['error'].idxmin()
    best_row = results_df.loc[best_idx]
    
    print(f"\n3. Most Accurate Threshold:")
    print(f"   {best_row['threshold_pct']:.0f}% (error: {best_row['error']:.1f}%)")
    
    # Diminishing returns analysis
    print("\n4. Diminishing Returns Analysis:")
    for i in range(len(results_df) - 1):
        current = results_df.iloc[i]
        next_row = results_df.iloc[i + 1]
        
        var_increase = next_row['total_selected'] - current['total_selected']
        impact_increase = next_row['coverage_pct'] - current['coverage_pct']
        
        if var_increase > 0:
            efficiency = impact_increase / var_increase
            print(f"   {current['threshold_pct']:.0f}% → {next_row['threshold_pct']:.0f}%: "
                  f"+{var_increase} vars for +{impact_increase:.1f}% impact "
                  f"(efficiency: {efficiency:.2f}% per var)")
    
    print("\n5. RECOMMENDATION:")
    
    # Check if 75% is reasonable
    row_75 = results_df[results_df['threshold_pct'] == 75].iloc[0]
    total_vars = row_75['total_selected']
    
    if row_75['error'] <= 5.0:  # Within 5% error
        print(f"   ✓ 75% threshold is REASONABLE:")
        print(f"     - Achieves {row_75['coverage_pct']:.1f}% coverage (close to target)")
        print(f"     - Requires {total_vars} variables (moderate)")
        print(f"     - Balances coverage with parsimony")
    else:
        print(f"   ⚠ Consider alternative threshold:")
        print(f"     - 75% has {row_75['error']:.1f}% error")
        print(f"     - Better option: {best_row['threshold_pct']:.0f}% ({best_row['error']:.1f}% error)")
    
    print(f"\n   Rule of thumb: 75% balances comprehensiveness (high coverage)")
    print(f"   with parsimony (avoiding too many variables)")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 4B")
    print("=" * 80)
    print()
    
    # Run sensitivity analysis
    results_df = run_sensitivity_analysis(
        method='avg_impact',
        thresholds=[0.60, 0.70, 0.75, 0.80, 0.90]
    )
    
    print("\n✓ Sensitivity analysis completed successfully!")
