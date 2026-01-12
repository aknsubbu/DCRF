"""
Experiment 2A: Value-Based Quadrant Division

This script divides variables into quadrants based on cumulative impact value,
not by count. Each quadrant contains variables that contribute to 25% of the
total impact.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_impact_values_from_exp1(results_path: str, method: str = 'avg_impact') -> Dict[str, float]:
    """
    Load impact values from Experiment 1A results.
    
    Parameters
    ----------
    results_path : str
        Path to experiment_1a_results.csv
    method : str
        Which method to use: 'linear_regression', 'partial_dependence', 
        'do_calculus', 'causal_forest', or 'avg_impact'
        
    Returns
    -------
    dict
        Dictionary of {variable: impact_value}
    """
    df = pd.read_csv(results_path)
    
    if method not in df.columns:
        raise ValueError(f"Method '{method}' not found. Available: {df.columns.tolist()}")
    
    impact_dict = dict(zip(df['variable'], df[method]))
    
    print(f"✓ Loaded {len(impact_dict)} variables from Experiment 1A")
    print(f"  Using method: {method}")
    
    return impact_dict


def divide_into_quadrants(impact_dict: Dict[str, float]) -> Dict[str, List[str]]:
    """
    Divide variables into quadrants based on cumulative impact value.
    
    Each quadrant contains variables that cumulatively contribute to 25% 
    of the total impact.
    
    Parameters
    ----------
    impact_dict : dict
        Dictionary of {variable: impact_value}
        
    Returns
    -------
    dict
        Dictionary with keys 'Q1', 'Q2', 'Q3', 'Q4', each containing list of variables
    """
    # Calculate total impact
    total_impact = sum(impact_dict.values())
    
    # Sort variables by impact (descending)
    sorted_vars = sorted(impact_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Define quadrant thresholds (cumulative percentages)
    thresholds = {
        'Q1': 0.25 * total_impact,
        'Q2': 0.50 * total_impact,
        'Q3': 0.75 * total_impact,
        'Q4': 1.00 * total_impact
    }
    
    # Assign variables to quadrants
    quadrants = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
    quadrant_impacts = {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0, 'Q4': 0.0}
    
    cumulative_impact = 0.0
    current_quadrant = 'Q1'
    
    for var, impact in sorted_vars:
        cumulative_impact += impact
        
        # Determine which quadrant this variable belongs to
        if cumulative_impact <= thresholds['Q1']:
            current_quadrant = 'Q1'
        elif cumulative_impact <= thresholds['Q2']:
            current_quadrant = 'Q2'
        elif cumulative_impact <= thresholds['Q3']:
            current_quadrant = 'Q3'
        else:
            current_quadrant = 'Q4'
        
        quadrants[current_quadrant].append(var)
        quadrant_impacts[current_quadrant] += impact
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUADRANT DIVISION SUMMARY")
    print("=" * 60)
    print(f"Total Impact: {total_impact:.4f}")
    print()
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        n_vars = len(quadrants[q])
        q_impact = quadrant_impacts[q]
        q_pct = (q_impact / total_impact) * 100
        
        print(f"{q}:")
        print(f"  Variables: {n_vars}")
        print(f"  Total Impact: {q_impact:.4f} ({q_pct:.1f}% of total)")
        if n_vars > 0:
            print(f"  Variables: {quadrants[q]}")
        print()
    
    return quadrants, quadrant_impacts


def save_quadrant_results(quadrants: Dict[str, List[str]], 
                          quadrant_impacts: Dict[str, float],
                          impact_dict: Dict[str, float],
                          method: str,
                          output_dir: str = 'results'):
    """
    Save quadrant division results to files.
    
    Parameters
    ----------
    quadrants : dict
        Quadrant assignments
    quadrant_impacts : dict
        Impact values per quadrant
    impact_dict : dict
        Original impact dictionary
    method : str
        Method name used
    output_dir : str
        Output directory
    """
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save quadrants as JSON
    json_path = output_path / f'quadrants_{method}.json'
    quadrant_data = {
        'method': method,
        'total_impact': sum(impact_dict.values()),
        'quadrants': quadrants,
        'quadrant_impacts': quadrant_impacts,
        'quadrant_counts': {q: len(vars_list) for q, vars_list in quadrants.items()}
    }
    
    with open(json_path, 'w') as f:
        json.dump(quadrant_data, f, indent=2)
    
    print(f"✓ Saved quadrant assignments: {json_path}")
    
    # Save as detailed CSV
    csv_path = output_path / f'quadrants_detailed_{method}.csv'
    
    rows = []
    for q, vars_list in quadrants.items():
        for var in vars_list:
            rows.append({
                'quadrant': q,
                'variable': var,
                'impact_value': impact_dict[var]
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['quadrant', 'impact_value'], ascending=[True, False])
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Saved detailed results: {csv_path}")
    
    # Save summary CSV
    summary_path = output_path / f'quadrants_summary_{method}.csv'
    
    total_impact = sum(impact_dict.values())
    summary_rows = []
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        summary_rows.append({
            'quadrant': q,
            'n_variables': len(quadrants[q]),
            'total_impact': quadrant_impacts[q],
            'pct_of_total': (quadrant_impacts[q] / total_impact) * 100
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    
    print(f"✓ Saved summary: {summary_path}")


def run_experiment_2a(exp1_results_path: str = None, methods: List[str] = None):
    """
    Run Experiment 2A: Value-Based Quadrant Division
    
    Parameters
    ----------
    exp1_results_path : str, optional
        Path to Experiment 1A results
    methods : list, optional
        List of methods to process
    """
    print("=" * 80)
    print("EXPERIMENT 2A: VALUE-BASED QUADRANT DIVISION")
    print("=" * 80)
    print()
    
    # Default path to Experiment 1A results
    if exp1_results_path is None:
        exp1_results_path = Path(__file__).parent.parent / 'experiment_1' / 'results' / 'experiment_1a_results.csv'
    
    # Default methods
    if methods is None:
        methods = ['avg_impact', 'linear_regression', 'partial_dependence', 'do_calculus', 'causal_forest']
    
    print(f"Loading results from: {exp1_results_path}")
    print()
    
    # Process each method
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Processing method: {method.upper()}")
        print('='*60)
        
        # Load impact values
        impact_dict = load_impact_values_from_exp1(exp1_results_path, method)
        
        # Divide into quadrants
        quadrants, quadrant_impacts = divide_into_quadrants(impact_dict)
        
        # Save results
        save_quadrant_results(quadrants, quadrant_impacts, impact_dict, method)
        
        # Store for comparison
        all_results[method] = {
            'quadrants': quadrants,
            'impacts': quadrant_impacts
        }
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2A COMPLETED")
    print("=" * 80)
    print(f"\nProcessed {len(methods)} methods successfully")
    
    return all_results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 2A")
    print("=" * 80)
    print()
    
    # Run experiment
    results = run_experiment_2a()
    
    print("\n✓ All quadrant divisions completed successfully!")
