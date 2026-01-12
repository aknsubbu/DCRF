"""
Experiment 1B: Impact Value Ranges with Confidence Intervals

This script extends Experiment 1A by calculating confidence intervals
for impact values using both bootstrap and analytical methods.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import load_data_and_pag, get_outcome_variable
from utils.pag_analysis import find_adjustment_set
from utils.effect_estimators import (
    LinearRegressionEstimator,
    PartialDependenceEstimator,
    DoCalculusEstimator,
    CausalForestEstimator
)


def bootstrap_confidence_interval(data: pd.DataFrame, treatment: str, outcome: str,
                                  adjustment_set: list, estimator_class, 
                                  n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence intervals for impact value.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment variable
    outcome : str
        Outcome variable
    adjustment_set : list
        Adjustment variables
    estimator_class : class
        Estimator class to use
    n_bootstrap : int
        Number of bootstrap iterations
    alpha : float
        Significance level (0.05 for 95% CI)
        
    Returns
    -------
    tuple
        (lower_bound, point_estimate, upper_bound)
    """
    n_samples = len(data)
    bootstrap_estimates = []
    
    # Calculate point estimate
    point_estimator = estimator_class()
    point_estimator.fit(data, treatment, outcome, adjustment_set)
    point_estimate = point_estimator.estimate_impact()
    
    # Bootstrap iterations
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_data = data.iloc[bootstrap_indices].reset_index(drop=True)
        
        try:
            # Fit estimator on bootstrap sample
            estimator = estimator_class()
            estimator.fit(bootstrap_data, treatment, outcome, adjustment_set)
            impact = estimator.estimate_impact()
            bootstrap_estimates.append(impact)
        except:
            # Skip if bootstrap sample causes issues
            continue
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    # Calculate percentiles
    lower_bound = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return (lower_bound, point_estimate, upper_bound)


def analytical_confidence_interval(data: pd.DataFrame, treatment: str, outcome: str,
                                   adjustment_set: list, estimator_class,
                                   alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Calculate analytical confidence intervals (when available).
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment variable  
    outcome : str
        Outcome variable
    adjustment_set : list
        Adjustment variables
    estimator_class : class
        Estimator class to use
    alpha : float
        Significance level
        
    Returns
    -------
    tuple
        (lower_bound, point_estimate, upper_bound)
    """
    estimator = estimator_class()
    estimator.fit(data, treatment, outcome, adjustment_set)
    point_estimate = estimator.estimate_impact()
    
    # Check if estimator has analytical CI method
    if hasattr(estimator, 'get_confidence_interval'):
        lower, upper = estimator.get_confidence_interval(alpha)
        return (lower, point_estimate, upper)
    else:
        # Return point estimate with no interval
        return (point_estimate, point_estimate, point_estimate)


def run_experiment_1b(n_bootstrap=1000, alpha=0.05, verbose=True):
    """
    Run Experiment 1B: Impact Value Ranges
    
    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations
    alpha : float
        Significance level for CIs
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    pd.DataFrame
        Results with confidence intervals
    """
    print("=" * 80)
    print("EXPERIMENT 1B: IMPACT VALUE RANGES WITH CONFIDENCE INTERVALS")
    print("=" * 80)
    print()
    
    # Load data and PAG
    if verbose:
        print("Step 1: Loading IHDP data and FCI PAG...")
    
    data, pag, var_names = load_data_and_pag()
    outcome_var = get_outcome_variable()
    
    # Variables to analyze
    variables_to_analyze = [v for v in var_names if v != outcome_var]
    
    if verbose:
        print(f"  Loaded {len(var_names)} variables")
        print(f"  Analyzing {len(variables_to_analyze)} variables")
        print()
        print(f"Step 2: Calculating confidence intervals ({n_bootstrap} bootstrap iterations)...")
        print()
    
    # Define estimator classes and methods
    methods = {
        'linear_regression': LinearRegressionEstimator,
        'partial_dependence': PartialDependenceEstimator,
        'do_calculus': DoCalculusEstimator,
        'causal_forest': CausalForestEstimator
    }
    
    # Results storage
    results = []
    
    # Process each variable and method
    total_tasks = len(variables_to_analyze) * len(methods)
    task_count = 0
    
    for var in variables_to_analyze:
        # Get adjustment set
        var_idx = var_names.index(var)
        outcome_idx = var_names.index(outcome_var)
        adjustment_indices = find_adjustment_set(pag, var_idx, outcome_idx)
        adjustment_vars = [var_names[idx] for idx in adjustment_indices]
        
        for method_name, estimator_class in methods.items():
            task_count += 1
            
            if verbose:
                print(f"[{task_count}/{total_tasks}] {var} - {method_name}")
            
            try:
                # Bootstrap CI
                if verbose:
                    print(f"  - Bootstrap CI... ", end='')
                boot_lower, boot_point, boot_upper = bootstrap_confidence_interval(
                    data, var, outcome_var, adjustment_vars, 
                    estimator_class, n_bootstrap, alpha
                )
                if verbose:
                    print(f"[{boot_lower:.4f}, {boot_point:.4f}, {boot_upper:.4f}]")
                
                # Analytical CI (if available)
                if verbose:
                    print(f"  - Analytical CI... ", end='')
                anal_lower, anal_point, anal_upper = analytical_confidence_interval(
                    data, var, outcome_var, adjustment_vars,
                    estimator_class, alpha
                )
                if verbose:
                    print(f"[{anal_lower:.4f}, {anal_point:.4f}, {anal_upper:.4f}]")
                
                # Store results
                results.append({
                    'variable': var,
                    'method': method_name,
                    'bootstrap_lower': boot_lower,
                    'bootstrap_point': boot_point,
                    'bootstrap_upper': boot_upper,
                    'analytical_lower': anal_lower,
                    'analytical_point': anal_point,
                    'analytical_upper': anal_upper,
                    'min_impact_value': boot_lower,  # Conservative estimate
                    'bootstrap_width': boot_upper - boot_lower,
                    'analytical_width': anal_upper - anal_lower
                })
                
                if verbose:
                    print()
                    
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Error: {str(e)}")
                    print()
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"\nSuccessfully processed {len(results_df)} variable-method combinations")
        print()
        
        # Compare bootstrap vs analytical
        print("Bootstrap vs Analytical CI Comparison:")
        print(f"  Average bootstrap CI width: {results_df['bootstrap_width'].mean():.4f}")
        print(f"  Average analytical CI width: {results_df['analytical_width'].mean():.4f}")
        
        # Show top variables by conservative estimate
        print(f"\nTop 10 variables by minimum (conservative) impact:")
        top_vars = results_df.groupby('variable')['min_impact_value'].mean().nlargest(10)
        print(top_vars)
    
    return results_df


def save_results(results_df, output_dir='results'):
    """
    Save results to CSV file.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    output_dir : str
        Output directory
    """
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    csv_path = output_path / 'experiment_1b_results.csv'
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Results saved to: {csv_path}")


def validate_results(results_df, verbose=True):
    """
    Validate confidence interval results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    verbose : bool
        Whether to print validation info
        
    Returns
    -------
    bool
        True if validation passes
    """
    if verbose:
        print("\n" + "=" * 80)
        print("VALIDATION CHECKS")
        print("=" * 80)
    
    checks_passed = True
    
    # Check 1: Lower <= Point <= Upper for bootstrap
    invalid_bootstrap = (
        (results_df['bootstrap_lower'] > results_df['bootstrap_point']) |
        (results_df['bootstrap_point'] > results_df['bootstrap_upper'])
    ).sum()
    
    if invalid_bootstrap > 0:
        if verbose:
            print(f"⚠ Warning: {invalid_bootstrap} invalid bootstrap CIs")
        checks_passed = False
    else:
        if verbose:
            print("✓ All bootstrap CIs are valid (lower <= point <= upper)")
    
    # Check 2: Lower <= Point <= Upper for analytical
    invalid_analytical = (
        (results_df['analytical_lower'] > results_df['analytical_point']) |
        (results_df['analytical_point'] > results_df['analytical_upper'])
    ).sum()
    
    if invalid_analytical > 0:
        if verbose:
            print(f"⚠ Warning: {invalid_analytical} invalid analytical CIs")
        checks_passed = False
    else:
        if verbose:
            print("✓ All analytical CIs are valid (lower <= point <= upper)")
    
    # Check 3: CI widths are reasonable
    max_width = results_df['bootstrap_width'].max()
    if max_width > 100:
        if verbose:
            print(f"⚠ Warning: Some CI widths are very large (max: {max_width:.2f})")
        checks_passed = False
    else:
        if verbose:
            print(f"✓ CI widths are reasonable (max: {max_width:.4f})")
    
    # Check 4: No negative minimum impacts
    negative_min = (results_df['min_impact_value'] < 0).sum()
    if negative_min > 0:
        if verbose:
            print(f"⚠ Warning: {negative_min} negative minimum impact values")
        checks_passed = False
    else:
        if verbose:
            print("✓ All minimum impact values are non-negative")
    
    if verbose:
        print("=" * 80)
        print()
    
    return checks_passed


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 1B")
    print("=" * 80)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run experiment with fewer bootstrap iterations for faster execution
    # Use n_bootstrap=1000 for final results
    results_df = run_experiment_1b(n_bootstrap=100, alpha=0.05, verbose=True)
    
    # Validate results
    validation_passed = validate_results(results_df, verbose=True)
    
    # Save results
    save_results(results_df)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1B COMPLETED")
    print("=" * 80)
    
    if validation_passed:
        print("✓ All validation checks passed!")
    else:
        print("⚠ Some validation checks failed. Please review results.")
    
    print("\nNote: This run used 100 bootstrap iterations for speed.")
    print("For final results, use n_bootstrap=1000 in the script.")
