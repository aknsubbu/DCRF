"""
Experiment 5C: Alternative Testing Strategy (Variance Contribution)

This script analyzes variance contribution of each variable using
model comparison (R² difference).
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_selected_variables() -> List[str]:
    """Load selected variables from Experiment 4."""
    csv_path = Path(__file__).parent.parent / 'experiment_4' / 'results' / 'selection_avg_impact_75pct.csv'
    df = pd.read_csv(csv_path)
    variables = df['variable'].tolist()
    print(f"✓ Loaded {len(variables)} selected variables")
    return variables


def load_ihdp_data() -> Tuple[pd.DataFrame, str]:
    """Load IHDP dataset and outcome variable."""
    from experiment_1.utils.data_loader import load_ihdp_data, get_outcome_variable
    data = load_ihdp_data()
    outcome = get_outcome_variable()
    return data, outcome


def calculate_variance_contribution(data: pd.DataFrame, variables: List[str],
                                   outcome: str, dropped_var: str = None) -> float:
    """
    Calculate R² for a model with specified variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    variables : list
        List of predictor variables
    outcome : str
        Outcome variable
    dropped_var : str, optional
        Variable being dropped (for logging)
        
    Returns
    -------
    float
        R² value
    """
    X = data[variables].values
    y = data[outcome].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    return r2


def test_variance_contribution_significance(data: pd.DataFrame,
                                           all_vars: List[str],
                                           outcome: str,
                                           var_to_test: str) -> Dict:
    """
    Test if variance contribution of a variable is significant.
    
    Uses F-test comparing full model vs reduced model.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    all_vars : list
        All predictor variables
    outcome : str
        Outcome variable
    var_to_test : str
        Variable to test
        
    Returns
    -------
    dict
        Test results
    """
    n = len(data)
    
    # Full model R²
    r2_full = calculate_variance_contribution(data, all_vars, outcome)
    
    # Reduced model R² (without var_to_test)
    reduced_vars = [v for v in all_vars if v != var_to_test]
    r2_reduced = calculate_variance_contribution(data, reduced_vars, outcome) if reduced_vars else 0
    
    # Variance contribution
    variance_contrib = r2_full - r2_reduced
    
    # F-test for significance
    # F = (R²_full - R²_reduced) / (1 - R²_full) * (n - k_full - 1) / (k_full - k_reduced)
    k_full = len(all_vars)
    k_reduced = len(reduced_vars)
    
    if 1 - r2_full > 0 and k_full > k_reduced:
        f_statistic = (variance_contrib / (1 - r2_full)) * (n - k_full - 1) / (k_full - k_reduced)
        p_value = 1 - stats.f.cdf(f_statistic, k_full - k_reduced, n - k_full - 1)
    else:
        f_statistic = np.nan
        p_value = np.nan
    
    result = {
        'variable': var_to_test,
        'r2_full': r2_full,
        'r2_reduced': r2_reduced,
        'variance_contribution': variance_contrib,
        'variance_contribution_pct': variance_contrib * 100,
        'f_statistic': f_statistic,
        'p_value': p_value,
        'is_significant': p_value < 0.05 if not np.isnan(p_value) else False,
        'n_samples': n,
        'n_predictors_full': k_full,
        'n_predictors_reduced': k_reduced
    }
    
    return result


def run_experiment_5c():
    """
    Run Experiment 5C: Variance Contribution Analysis
    """
    print("=" * 80)
    print("EXPERIMENT 5C: VARIANCE CONTRIBUTION ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    selected_vars = load_selected_variables()
    data, outcome = load_ihdp_data()
    
    print(f"Outcome variable: {outcome}")
    print(f"Analyzing {len(selected_vars)} variables...")
    print()
    
    # Calculate full model R²
    print("Step 1: Fitting full model...")
    r2_full = calculate_variance_contribution(data, selected_vars, outcome)
    print(f"  Full model R²: {r2_full:.4f}")
    print()
    
    # Test each variable
    print("Step 2: Testing variance contribution for each variable...")
    print()
    
    results = []
    
    for var in selected_vars:
        print(f"  {var}:")
        
        test_result = test_variance_contribution_significance(
            data, selected_vars, outcome, var
        )
        results.append(test_result)
        
        print(f"    Variance contribution: {test_result['variance_contribution']:.4f} ({test_result['variance_contribution_pct']:.2f}%)")
        print(f"    F-statistic: {test_result['f_statistic']:.4f}")
        print(f"    P-value: {test_result['p_value']:.4f} {'*' if test_result['is_significant'] else ''}")
        print()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('variance_contribution', ascending=False)
    
    # Summary
    print("=" * 60)
    print("VARIANCE CONTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"\nFull model R²: {r2_full:.4f}")
    
    sig_count = results_df['is_significant'].sum()
    print(f"Significant contributions (p < 0.05): {sig_count}/{len(results_df)}")
    
    print(f"\nTop 5 variables by variance contribution:")
    top5 = results_df.head(5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        sig_mark = '*' if row['is_significant'] else ''
        print(f"  {i}. {row['variable']}: {row['variance_contribution']:.4f} "
              f"({row['variance_contribution_pct']:.2f}%) {sig_mark}")
    
    # Save results
    output_path = Path(__file__).parent / 'results'
    
    csv_path = output_path / 'variance_contributions.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results: {csv_path}")
    
    json_path = output_path / 'variance_contributions.json'
    results_list = results_df.to_dict('records')
    # Convert numpy types
    results_json = []
    for r in results_list:
        r_copy = r.copy()
        for key, val in r_copy.items():
            if isinstance(val, (np.bool_, np.integer, np.floating)):
                r_copy[key] = val.item()
            elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                r_copy[key] = None
        results_json.append(r_copy)
    
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Saved JSON: {json_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 5C COMPLETED")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 5C")
    print("=" * 80)
    print()
    
    # Run experiment
    results_df = run_experiment_5c()
    
    print("\n✓ Variance contribution analysis completed successfully!")
