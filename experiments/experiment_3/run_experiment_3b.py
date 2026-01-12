"""
Experiment 3B: Cross-Reference with Quadrants

This script creates a matrix showing the distribution of variable types
(Independent, Edge, Confounder, Other) across quadrants (Q1-Q4).
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

# Set style
sns.set_style("whitegrid")


def load_variable_roles(results_dir: str = 'results') -> Dict[str, str]:
    """
    Load variable role classifications from Experiment 3A.
    
    Parameters
    ----------
    results_dir : str
        Results directory
        
    Returns
    -------
    dict
        Variable roles dictionary
    """
    json_path = Path(__file__).parent / results_dir / 'variable_roles.json'
    
    with open(json_path, 'r') as f:
        role_dict = json.load(f)
    
    print(f"✓ Loaded role classifications for {len(role_dict)} variables")
    
    return role_dict


def load_quadrant_assignments(exp2_results_dir: str, method: str) -> Dict[str, List[str]]:
    """
    Load quadrant assignments from Experiment 2.
    
    Parameters
    ----------
    exp2_results_dir : str
        Experiment 2 results directory
    method : str
        Method name
        
    Returns
    -------
    dict
        Quadrant assignments
    """
    json_path = Path(exp2_results_dir) / f'quadrants_{method}.json'
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded quadrant assignments for method: {method}")
    
    return data['quadrants']


def create_crosstab_matrix(role_dict: Dict[str, str], quadrants: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Create cross-tabulation matrix of Quadrants × Variable Roles.
    
    Parameters
    ----------
    role_dict : dict
        Variable role classifications
    quadrants : dict
        Quadrant assignments
        
    Returns
    -------
    pd.DataFrame
        Cross-tabulation matrix
    """
    # Create data for matrix
    matrix_data = {
        'independent': {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']},
        'confounder': {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']},
        'edge': {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']},
        'other': {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']},
        'treatment': {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']},
        'outcome': {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    }
    
    # Fill matrix
    for quadrant, vars_list in quadrants.items():
        for var in vars_list:
            if var in role_dict:
                role = role_dict[var]
                matrix_data[role][quadrant].append(var)
    
    # Convert to DataFrame with counts
    count_data = []
    for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
        row = {
            'Quadrant': quadrant,
            'Independent': len(matrix_data['independent'][quadrant]),
            'Confounder': len(matrix_data['confounder'][quadrant]),
            'Edge': len(matrix_data['edge'][quadrant]),
            'Other': len(matrix_data['other'][quadrant]),
            'Treatment': len(matrix_data['treatment'][quadrant]),
            'Outcome': len(matrix_data['outcome'][quadrant])
        }
        count_data.append(row)
    
    count_df = pd.DataFrame(count_data)
    count_df = count_df.set_index('Quadrant')
    
    return count_df, matrix_data


def create_detailed_matrix(matrix_data: Dict) -> pd.DataFrame:
    """
    Create detailed matrix with variable names (not just counts).
    
    Parameters
    ----------
    matrix_data : dict
        Matrix data with variable lists
        
    Returns
    -------
    pd.DataFrame
        Detailed matrix
    """
    rows = []
    
    for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
        row = {
            'Quadrant': quadrant,
            'Independent': ', '.join(matrix_data['independent'][quadrant]) or '-',
            'Confounder': ', '.join(matrix_data['confounder'][quadrant]) or '-',
            'Edge': ', '.join(matrix_data['edge'][quadrant]) or '-',
            'Other': ', '.join(matrix_data['other'][quadrant]) or '-',
            'Treatment': ', '.join(matrix_data['treatment'][quadrant]) or '-',
            'Outcome': ', '.join(matrix_data['outcome'][quadrant]) or '-'
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def visualize_crosstab(count_df: pd.DataFrame, method: str, output_dir: str = 'results'):
    """
    Create visualization of the cross-tabulation.
    
    Parameters
    ----------
    count_df : pd.DataFrame
        Count matrix
    method : str
        Method name
    output_dir : str
        Output directory
    """
    output_path = Path(__file__).parent / output_dir / 'figures'
    output_path.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Heatmap of counts
    ax = axes[0]
    
    # Select main role types (exclude treatment/outcome for clarity)
    plot_df = count_df[['Independent', 'Confounder', 'Edge', 'Other']]
    
    sns.heatmap(plot_df.T, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Variables'},
                ax=ax, linewidths=1, linecolor='black')
    
    ax.set_xlabel('Quadrant', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable Role', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Variable Roles Across Quadrants', 
                 fontsize=13, fontweight='bold')
    
    # 2. Stacked bar chart
    ax = axes[1]
    
    plot_df.T.plot(kind='bar', stacked=True, ax=ax, 
                   color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'],
                   width=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Variable Role', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Variables', fontsize=12, fontweight='bold')
    ax.set_title('Variable Counts by Role and Quadrant', fontsize=13, fontweight='bold')
    ax.legend(title='Quadrant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.suptitle(f'Quadrant-Role Cross-Tabulation - {method.upper()}', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_path / f'crosstab_roles_quadrants_{method}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {fig_path}")
    
    plt.show()


def create_role_distribution_plot(role_dict: Dict[str, str], output_dir: str = 'results'):
    """
    Create overall distribution plot of variable roles.
    
    Parameters
    ----------
    role_dict : dict
        Variable roles
    output_dir : str
        Output directory
    """
    output_path = Path(__file__).parent / output_dir / 'figures'
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Count roles
    role_counts = {}
    for role in role_dict.values():
        role_counts[role] = role_counts.get(role, 0) + 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    roles = list(role_counts.keys())
    counts = list(role_counts.values())
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c'][:len(roles)]
    
    bars = ax.bar(roles, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_xlabel('Variable Role', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Variables', fontsize=13, fontweight='bold')
    ax.set_title('Overall Distribution of Variable Roles in PAG', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save
    fig_path = output_path / 'role_distribution.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved role distribution plot: {fig_path}")
    
    plt.show()


def run_experiment_3b(method: str = 'avg_impact'):
    """
    Run Experiment 3B: Cross-Reference with Quadrants
    
    Parameters
    ----------
    method : str
        Method to use from Experiment 2
    """
    print("=" * 80)
    print("EXPERIMENT 3B: CROSS-REFERENCE WITH QUADRANTS")
    print("=" * 80)
    print()
    
    # Load variable roles from Experiment 3A
    print("Step 1: Loading variable roles...")
    role_dict = load_variable_roles()
    print()
    
    # Create role distribution plot
    print("Step 2: Creating overall role distribution plot...")
    create_role_distribution_plot(role_dict)
    print()
    
    # Load quadrant assignments from Experiment 2
    print(f"Step 3: Loading quadrant assignments (method: {method})...")
    exp2_results = Path(__file__).parent.parent / 'experiment_2' / 'results'
    quadrants = load_quadrant_assignments(exp2_results, method)
    print()
    
    # Create cross-tabulation
    print("Step 4: Creating cross-tabulation matrix...")
    count_df, matrix_data = create_crosstab_matrix(role_dict, quadrants)
    
    print("\nCount Matrix (Quadrants × Roles):")
    print(count_df)
    print()
    
    # Create detailed matrix
    detailed_df = create_detailed_matrix(matrix_data)
    
    print("Detailed Matrix (with variable names):")
    print(detailed_df.to_string(index=False))
    print()
    
    # Save results
    output_path = Path(__file__).parent / 'results'
    output_path.mkdir(exist_ok=True, parents=True)
    
    count_csv = output_path / f'crosstab_counts_{method}.csv'
    count_df.to_csv(count_csv)
    print(f"✓ Saved count matrix: {count_csv}")
    
    detailed_csv = output_path / f'crosstab_detailed_{method}.csv'
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"✓ Saved detailed matrix: {detailed_csv}")
    
    # Save as JSON
    json_path = output_path / f'crosstab_{method}.json'
    crosstab_data = {
        'method': method,
        'counts': count_df.to_dict(),
        'variables': matrix_data
    }
    with open(json_path, 'w') as f:
        json.dump(crosstab_data, f, indent=2)
    print(f"✓ Saved JSON: {json_path}")
    
    # Visualize
    print("\nStep 5: Creating visualizations...")
    visualize_crosstab(count_df, method)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 3B COMPLETED")
    print("=" * 80)
    
    return count_df, detailed_df, matrix_data


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 3B")
    print("=" * 80)
    print()
    
    # Run experiment
    count_df, detailed_df, matrix_data = run_experiment_3b(method='avg_impact')
    
    print("\n✓ Cross-reference analysis completed successfully!")
