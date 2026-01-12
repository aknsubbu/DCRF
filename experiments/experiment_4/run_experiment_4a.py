"""
Experiment 4A: Equal Count Selection (75% Rule)

This script selects equal numbers of variables from Q1, Q2, Q3 such that
they cumulatively cover 75% of total impact (Q4 excluded).
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_quadrant_data(exp2_results_dir: str, method: str) -> Tuple[Dict, float]:
    """
    Load quadrant assignments and calculate total impact.
    
    Parameters
    ----------
    exp2_results_dir : str
        Experiment 2 results directory
    method : str
        Method name
        
    Returns
    -------
    tuple
        (quadrant_data dict, total_impact float)
    """
    json_path = Path(exp2_results_dir) / f'quadrants_{method}.json'
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded quadrant data for method: {method}")
    print(f"  Total impact: {data['total_impact']:.4f}")
    
    return data, data['total_impact']


def load_variable_impacts(exp1_results_path: str, method: str) -> Dict[str, float]:
    """
    Load individual variable impact values.
    
    Parameters
    ----------
    exp1_results_path : str
        Path to Experiment 1A results
    method : str
        Method column name
        
    Returns
    -------
    dict
        {variable: impact_value}
    """
    df = pd.read_csv(exp1_results_path)
    
    # Handle avg_impact vs method-specific columns
    if method == 'avg_impact':
        impact_dict = dict(zip(df['variable'], df['avg_impact']))
    else:
        impact_dict = dict(zip(df['variable'], df[method]))
    
    return impact_dict


def select_equal_count_for_target(quadrants: Dict, impact_dict: Dict[str, float],
                                  target_impact: float, total_impact: float) -> Dict:
    """
    Select equal number of variables from Q1, Q2, Q3 to reach target impact.
    
    Parameters
    ----------
    quadrants : dict
        Quadrant assignments
    impact_dict : dict
        Variable impact values
    target_impact : float
        Target cumulative impact
    total_impact : float
        Total impact of all variables
        
    Returns
    -------
    dict
        Selection results
    """
    # Get variables and impacts for Q1, Q2, Q3 (exclude Q4)
    q1_vars = quadrants['Q1']
    q2_vars = quadrants['Q2']
    q3_vars = quadrants['Q3']
    
    # Sort each quadrant by impact (descending)
    q1_sorted = sorted([(v, impact_dict[v]) for v in q1_vars], 
                      key=lambda x: x[1], reverse=True)
    q2_sorted = sorted([(v, impact_dict[v]) for v in q2_vars],
                      key=lambda x: x[1], reverse=True)
    q3_sorted = sorted([(v, impact_dict[v]) for v in q3_vars],
                      key=lambda x: x[1], reverse=True)
    
    print(f"\nAvailable variables by quadrant:")
    print(f"  Q1: {len(q1_vars)} variables")
    print(f"  Q2: {len(q2_vars)} variables")
    print(f"  Q3: {len(q3_vars)} variables")
    print(f"  Q4: {len(quadrants['Q4'])} variables (EXCLUDED)")
    
    # Try different counts to find closest to target
    max_count = min(len(q1_vars), len(q2_vars), len(q3_vars))
    
    best_selection = None
    best_diff = float('inf')
    
    print(f"\nTrying different selection counts (equal from each quadrant):")
    
    for count in range(1, max_count + 1):
        # Select top 'count' from each quadrant
        q1_selected = q1_sorted[:count]
        q2_selected = q2_sorted[:count]
        q3_selected = q3_sorted[:count]
        
        # Calculate cumulative impact
        cumulative = sum(v[1] for v in q1_selected + q2_selected + q3_selected)
        coverage_pct = (cumulative / total_impact) * 100
        diff = abs(cumulative - target_impact)
        
        print(f"  Count={count}: Impact={cumulative:.4f} ({coverage_pct:.1f}% of total), "
              f"Diff from target={diff:.4f}")
        
        if diff < best_diff:
            best_diff = diff
            best_selection = {
                'count_per_quadrant': count,
                'total_selected': count * 3,
                'q1_selected': q1_selected,
                'q2_selected': q2_selected,
                'q3_selected': q3_selected,
                'cumulative_impact': cumulative,
                'coverage_pct': coverage_pct,
                'target_achieved': cumulative >= target_impact
            }
    
    return best_selection


def run_experiment_4a(method: str = 'avg_impact', target_pct: float = 0.75):
    """
    Run Experiment 4A: Equal Count Selection for 75% Rule
    
    Parameters
    ----------
    method : str
        Method from Experiment 1/2
    target_pct : float
        Target percentage of total impact (default: 0.75 for 75%)
    """
    print("=" * 80)
    print("EXPERIMENT 4A: EQUAL COUNT SELECTION (75% RULE)")
    print("=" * 80)
    print()
    
    # Load data
    exp2_results = Path(__file__).parent.parent / 'experiment_2' / 'results'
    exp1_results = Path(__file__).parent.parent / 'experiment_1' / 'results' / 'experiment_1a_results.csv'
    
    print(f"Method: {method}")
    print(f"Target: {target_pct * 100}% of total impact")
    print()
    
    quadrant_data, total_impact = load_quadrant_data(exp2_results, method)
    impact_dict = load_variable_impacts(exp1_results, method)
    
    target_impact = target_pct * total_impact
    print(f"\nTarget impact: {target_impact:.4f} ({target_pct * 100}% of {total_impact:.4f})")
    
    # Perform selection
    selection = select_equal_count_for_target(
        quadrant_data['quadrants'],
        impact_dict,
        target_impact,
        total_impact
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("SELECTION RESULTS")
    print("=" * 60)
    print(f"Variables per quadrant: {selection['count_per_quadrant']}")
    print(f"Total variables selected: {selection['total_selected']}")
    print(f"Cumulative impact: {selection['cumulative_impact']:.4f}")
    print(f"Coverage: {selection['coverage_pct']:.1f}% of total impact")
    print(f"Target achieved: {'✓ YES' if selection['target_achieved'] else '✗ NO'}")
    print()
    
    print("Selected variables:")
    print(f"\n  From Q1 ({selection['count_per_quadrant']} vars):")
    for var, impact in selection['q1_selected']:
        print(f"    {var}: {impact:.4f}")
    
    print(f"\n  From Q2 ({selection['count_per_quadrant']} vars):")
    for var, impact in selection['q2_selected']:
        print(f"    {var}: {impact:.4f}")
    
    print(f"\n  From Q3 ({selection['count_per_quadrant']} vars):")
    for var, impact in selection['q3_selected']:
        print(f"    {var}: {impact:.4f}")
    
    # Save results
    output_path = Path(__file__).parent / 'results'
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create summary
    all_selected = (selection['q1_selected'] + selection['q2_selected'] + 
                   selection['q3_selected'])
    
    summary = {
        'method': method,
        'target_pct': target_pct,
        'target_impact': target_impact,
        'total_impact': total_impact,
        'count_per_quadrant': selection['count_per_quadrant'],
        'total_selected': selection['total_selected'],
        'cumulative_impact': selection['cumulative_impact'],
        'coverage_pct': selection['coverage_pct'],
        'target_achieved': selection['target_achieved'],
        'selected_variables': {
            'Q1': [v for v, _ in selection['q1_selected']],
            'Q2': [v for v, _ in selection['q2_selected']],
            'Q3': [v for v, _ in selection['q3_selected']]
        }
    }
    
    # Save as JSON
    json_path = output_path / f'selection_{method}_{int(target_pct*100)}pct.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved selection: {json_path}")
    
    # Save as CSV
    csv_data = []
    for var, impact in all_selected:
        quadrant = None
        if var in summary['selected_variables']['Q1']:
            quadrant = 'Q1'
        elif var in summary['selected_variables']['Q2']:
            quadrant = 'Q2'
        else:
            quadrant = 'Q3'
        
        csv_data.append({
            'variable': var,
            'quadrant': quadrant,
            'impact_value': impact
        })
    
    df = pd.DataFrame(csv_data)
    df = df.sort_values(['quadrant', 'impact_value'], ascending=[True, False])
    
    csv_path = output_path / f'selection_{method}_{int(target_pct*100)}pct.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved selection CSV: {csv_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 4A COMPLETED")
    print("=" * 80)
    
    return selection, summary


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 4A")
    print("=" * 80)
    print()
    
    # Run with default 75% target
    selection, summary = run_experiment_4a(method='avg_impact', target_pct=0.75)
    
    print("\n✓ Variable selection completed successfully!")
