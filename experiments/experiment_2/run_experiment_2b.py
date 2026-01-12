"""
Experiment 2B: Validate Quadrant Distribution

This script validates that the quadrant division correctly distributes
impact values and visualizes the results.
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
sns.set_style("whitegrid")


def load_quadrant_results(results_dir: str, method: str) -> Dict:
    """
    Load quadrant results from Experiment 2A.
    
    Parameters
    ----------
    results_dir : str
        Results directory
    method : str
        Method name
        
    Returns
    -------
    dict
        Quadrant results
    """
    json_path = Path(results_dir) / f'quadrants_{method}.json'
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded quadrant results for method: {method}")
    
    return data


def validate_quadrant_distribution(quadrant_data: Dict, method: str, tolerance: float = 2.0) -> bool:
    """
    Validate that each quadrant has approximately 25% of total impact.
    
    Parameters
    ----------
    quadrant_data : dict
        Quadrant data from Experiment 2A
    method : str
        Method name
    tolerance : float
        Acceptable deviation from 25% (in percentage points)
        
    Returns
    -------
    bool
        True if validation passes
    """
    print("\n" + "=" * 60)
    print(f"VALIDATION: {method.upper()}")
    print("=" * 60)
    
    total_impact = quadrant_data['total_impact']
    quadrant_impacts = quadrant_data['quadrant_impacts']
    quadrant_counts = quadrant_data['quadrant_counts']
    
    print(f"Total Impact: {total_impact:.4f}")
    print()
    
    all_valid = True
    expected_pct = 25.0
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_impact = quadrant_impacts[q]
        q_pct = (q_impact / total_impact) * 100
        q_count = quadrant_counts[q]
        
        # Check if within tolerance
        deviation = abs(q_pct - expected_pct)
        is_valid = deviation <= tolerance
        
        status = "✓" if is_valid else "✗"
        
        print(f"{status} {q}:")
        print(f"    Impact: {q_impact:.4f} ({q_pct:.2f}% of total)")
        print(f"    Expected: ~{expected_pct}%")
        print(f"    Deviation: {deviation:.2f} percentage points")
        print(f"    Variables: {q_count}")
        
        if not is_valid:
            print(f"    ⚠ WARNING: Exceeds tolerance of {tolerance}%")
            all_valid = False
        
        print()
    
    # Overall validation
    total_pct = sum((quadrant_impacts[q] / total_impact) * 100 for q in ['Q1', 'Q2', 'Q3', 'Q4'])
    print(f"Total percentage: {total_pct:.2f}% (should be 100%)")
    
    if abs(total_pct - 100.0) > 0.1:
        print("⚠ WARNING: Total percentage does not sum to 100%")
        all_valid = False
    
    print("=" * 60)
    
    if all_valid:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
    
    return all_valid


def visualize_quadrant_distribution(quadrant_data: Dict, method: str, output_dir: str = 'results'):
    """
    Create visualizations for quadrant distribution.
    
    Parameters
    ----------
    quadrant_data : dict
        Quadrant data
    method : str
        Method name
    output_dir : str
        Output directory for figures
    """
    output_path = Path(__file__).parent / output_dir / 'figures'
    output_path.mkdir(exist_ok=True, parents=True)
    
    total_impact = quadrant_data['total_impact']
    quadrant_impacts = quadrant_data['quadrant_impacts']
    quadrant_counts = quadrant_data['quadrant_counts']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Bar chart of impact distribution
    ax = axes[0, 0]
    quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
    impacts = [quadrant_impacts[q] for q in quadrants]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    bars = ax.bar(quadrants, impacts, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=total_impact * 0.25, color='red', linestyle='--', 
               label='Expected (25%)', linewidth=2)
    ax.set_ylabel('Total Impact', fontsize=12, fontweight='bold')
    ax.set_title('Impact Distribution by Quadrant', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, impact in zip(bars, impacts):
        height = bar.get_height()
        pct = (impact / total_impact) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{impact:.2f}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Variable count per quadrant
    ax = axes[0, 1]
    counts = [quadrant_counts[q] for q in quadrants]
    
    bars = ax.bar(quadrants, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Variables', fontsize=12, fontweight='bold')
    ax.set_title('Variable Count by Quadrant', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 3. Cumulative impact curve
    ax = axes[1, 0]
    
    # Load detailed data for cumulative curve
    csv_path = Path(__file__).parent / output_dir / f'quadrants_detailed_{method}.csv'
    df_detailed = pd.read_csv(csv_path)
    
    # Sort by impact descending
    df_detailed = df_detailed.sort_values('impact_value', ascending=False)
    df_detailed['cumulative_impact'] = df_detailed['impact_value'].cumsum()
    df_detailed['cumulative_pct'] = (df_detailed['cumulative_impact'] / total_impact) * 100
    
    ax.plot(range(1, len(df_detailed) + 1), df_detailed['cumulative_pct'], 
            marker='o', linewidth=2, markersize=4, color='#2c3e50')
    
    # Add quadrant boundaries
    ax.axhline(y=25, color='#2ecc71', linestyle='--', alpha=0.7, label='Q1 (25%)')
    ax.axhline(y=50, color='#3498db', linestyle='--', alpha=0.7, label='Q2 (50%)')
    ax.axhline(y=75, color='#f39c12', linestyle='--', alpha=0.7, label='Q3 (75%)')
    
    ax.set_xlabel('Variable Rank (by impact)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Impact (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Impact Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 4. Pie chart of impact distribution
    ax = axes[1, 1]
    
    ax.pie(impacts, labels=quadrants, autopct='%1.1f%%', colors=colors,
           startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    ax.set_title('Impact Distribution (Percentage)', fontsize=13, fontweight='bold')
    
    # Overall title
    fig.suptitle(f'Quadrant Distribution Analysis - {method.upper()}', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_path / f'quadrant_distribution_{method}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {fig_path}")
    
    plt.show()


def create_comparison_visualization(results_dir: str, methods: List[str], output_dir: str = 'results'):
    """
    Create comparison visualization across methods.
    
    Parameters
    ----------
    results_dir : str
        Results directory
    methods : list
        List of methods to compare
    output_dir : str
        Output directory
    """
    output_path = Path(__file__).parent / output_dir / 'figures'
    output_path.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data
    quadrant_data = []
    
    for method in methods:
        data = load_quadrant_results(results_dir, method)
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            quadrant_data.append({
                'method': method,
                'quadrant': q,
                'n_variables': data['quadrant_counts'][q],
                'total_impact': data['quadrant_impacts'][q],
                'pct_impact': (data['quadrant_impacts'][q] / data['total_impact']) * 100
            })
    
    df = pd.DataFrame(quadrant_data)
    
    # Plot 1: Variable counts comparison
    ax = axes[0]
    pivot_counts = df.pivot(index='quadrant', columns='method', values='n_variables')
    pivot_counts.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Quadrant', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Variables', fontsize=12, fontweight='bold')
    ax.set_title('Variable Count Comparison Across Methods', fontsize=13, fontweight='bold')
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # Plot 2: Impact percentage comparison
    ax = axes[1]
    pivot_pct = df.pivot(index='quadrant', columns='method', values='pct_impact')
    pivot_pct.plot(kind='bar', ax=ax, width=0.8)
    ax.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Expected (25%)')
    ax.set_xlabel('Quadrant', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Total Impact', fontsize=12, fontweight='bold')
    ax.set_title('Impact Distribution Comparison Across Methods', fontsize=13, fontweight='bold')
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    
    fig_path = output_path / 'quadrant_comparison_all_methods.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison visualization: {fig_path}")
    
    plt.show()


def run_experiment_2b(results_dir: str = None, methods: List[str] = None):
    """
    Run Experiment 2B: Validate Quadrant Distribution
    
    Parameters
    ----------
    results_dir : str, optional
        Results directory from Experiment 2A
    methods : list, optional
        List of methods to validate
    """
    print("=" * 80)
    print("EXPERIMENT 2B: VALIDATE QUADRANT DISTRIBUTION")
    print("=" * 80)
    print()
    
    # Default results directory
    if results_dir is None:
        results_dir = Path(__file__).parent / 'results'
    
    # Default methods
    if methods is None:
        methods = ['avg_impact', 'linear_regression', 'partial_dependence', 'do_calculus', 'causal_forest']
    
    validation_results = {}
    
    # Validate each method
    for method in methods:
        try:
            # Load results
            quadrant_data = load_quadrant_results(results_dir, method)
            
            # Validate distribution
            is_valid = validate_quadrant_distribution(quadrant_data, method)
            validation_results[method] = is_valid
            
            # Visualize
            print(f"\nCreating visualizations for {method}...")
            visualize_quadrant_distribution(quadrant_data, method)
            
        except Exception as e:
            print(f"✗ Error processing {method}: {str(e)}")
            validation_results[method] = False
    
    # Create comparison visualization
    print("\nCreating comparison visualization across all methods...")
    create_comparison_visualization(results_dir, methods)
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 2B COMPLETED")
    print("=" * 80)
    print("\nValidation Summary:")
    for method, is_valid in validation_results.items():
        status = "✓ PASSED" if is_valid else "✗ FAILED"
        print(f"  {method}: {status}")
    
    return validation_results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 2B")
    print("=" * 80)
    print()
    
    # Run validation
    results = run_experiment_2b()
    
    print("\n✓ Validation and visualization completed!")
