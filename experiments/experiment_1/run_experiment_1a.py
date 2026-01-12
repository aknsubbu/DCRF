"""
Experiment 1A: Direct Causal Effect Estimation

This script calculates impact values for each variable in the IHDP dataset
using multiple causal effect estimation methods:
1. Linear Regression (standardized coefficients)
2. Partial Dependence Plots (slope)
3. Do-calculus (backdoor adjustment)
4. Causal Forest (variable importance proxy)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import load_data_and_pag, get_outcome_variable, get_treatment_variable
from utils.pag_analysis import find_adjustment_set, summarize_pag_structure
from utils.effect_estimators import (
    LinearRegressionEstimator,
    PartialDependenceEstimator,
    DoCalculusEstimator,
    CausalForestEstimator
)


def run_experiment_1a(verbose=True):
    """
    Run Experiment 1A: Direct Causal Effect Estimation
    
    Parameters
    ----------
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    pd.DataFrame
        Results with columns: variable, linear_regression, partial_dependence, 
        do_calculus, causal_forest
    """
    print("=" * 80)
    print("EXPERIMENT 1A: DIRECT CAUSAL EFFECT ESTIMATION")
    print("=" * 80)
    print()
    
    # Load data and PAG
    if verbose:
        print("Step 1: Loading IHDP data and FCI PAG...")
    
    data, pag, var_names = load_data_and_pag()
    outcome_var = get_outcome_variable()
    
    if verbose:
        print(f"  Loaded {len(var_names)} variables")
        print(f"  Outcome variable: {outcome_var}")
        print()
        
        print("Step 2: Analyzing PAG structure...")
        summarize_pag_structure(pag, var_names)
        print()
    
    # Variables to analyze (all except outcome)
    variables_to_analyze = [v for v in var_names if v != outcome_var]
    
    if verbose:
        print(f"Step 3: Estimating causal effects for {len(variables_to_analyze)} variables...")
        print()
    
    # Results storage
    results = {
        'variable': [],
        'linear_regression': [],
        'partial_dependence': [],
        'do_calculus': [],
        'causal_forest': []
    }
    
    # Process each variable
    for i, var in enumerate(variables_to_analyze):
        if verbose:
            print(f"[{i+1}/{len(variables_to_analyze)}] Processing variable: {var}")
        
        # Get variable index
        var_idx = var_names.index(var)
        outcome_idx = var_names.index(outcome_var)
        
        # Find adjustment set from PAG
        adjustment_indices = find_adjustment_set(
            pag, var_idx, outcome_idx, var_names if verbose else None
        )
        adjustment_vars = [var_names[idx] for idx in adjustment_indices]
        
        # Skip variables that are not in the dataset or cause issues
        try:
            # Method 1: Linear Regression
            if verbose:
                print(f"  - Linear Regression... ", end='')
            lr_est = LinearRegressionEstimator()
            lr_est.fit(data, var, outcome_var, adjustment_vars)
            lr_impact = lr_est.estimate_impact()
            if verbose:
                print(f"Impact: {lr_impact:.4f}")
            
            # Method 2: Partial Dependence
            if verbose:
                print(f"  - Partial Dependence... ", end='')
            pd_est = PartialDependenceEstimator(model_type='rf', n_estimators=100)
            pd_est.fit(data, var, outcome_var, adjustment_vars)
            pd_impact = pd_est.estimate_impact()
            if verbose:
                print(f"Impact: {pd_impact:.4f}")
            
            # Method 3: Do-calculus
            if verbose:
                print(f"  - Do-calculus... ", end='')
            do_est = DoCalculusEstimator(n_bins=5)
            do_est.fit(data, var, outcome_var, adjustment_vars)
            do_impact = do_est.estimate_impact()
            if verbose:
                print(f"Impact: {do_impact:.4f}")
            
            # Method 4: Causal Forest (RF proxy)
            if verbose:
                print(f"  - Causal Forest (RF proxy)... ", end='')
            cf_est = CausalForestEstimator()
            cf_est.fit(data, var, outcome_var, adjustment_vars)
            cf_impact = cf_est.estimate_impact()
            if verbose:
                print(f"Impact: {cf_impact:.4f}")
            
            # Store results
            results['variable'].append(var)
            results['linear_regression'].append(lr_impact)
            results['partial_dependence'].append(pd_impact)
            results['do_calculus'].append(do_impact)
            results['causal_forest'].append(cf_impact)
            
            if verbose:
                print()
                
        except Exception as e:
            if verbose:
                print(f"  ⚠ Error processing {var}: {str(e)}")
                print()
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"\nSuccessfully processed {len(results_df)} variables")
        print(f"\nTop 10 variables by average impact across methods:")
        print()
        
        # Calculate average impact
        results_df['avg_impact'] = results_df[[
            'linear_regression', 'partial_dependence', 
            'do_calculus', 'causal_forest'
        ]].mean(axis=1)
        
        top_10 = results_df.nlargest(10, 'avg_impact')
        print(top_10[['variable', 'avg_impact', 'linear_regression', 
                      'partial_dependence', 'do_calculus', 'causal_forest']])
    
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
    
    csv_path = output_path / 'experiment_1a_results.csv'
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Results saved to: {csv_path}")


def validate_results(results_df, verbose=True):
    """
    Validate that results are reasonable.
    
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
    
    # Check 1: No NaN values
    nan_count = results_df.isnull().sum().sum()
    if nan_count > 0:
        if verbose:
            print(f"⚠ Warning: Found {nan_count} NaN values")
        checks_passed = False
    else:
        if verbose:
            print("✓ No NaN values found")
    
    # Check 2: All impacts are non-negative
    method_cols = ['linear_regression', 'partial_dependence', 'do_calculus', 'causal_forest']
    negative_count = (results_df[method_cols] < 0).sum().sum()
    if negative_count > 0:
        if verbose:
            print(f"⚠ Warning: Found {negative_count} negative impact values")
        checks_passed = False
    else:
        if verbose:
            print("✓ All impact values are non-negative")
    
    # Check 3: Reasonable magnitude
    max_impact = results_df[method_cols].max().max()
    if max_impact > 100:
        if verbose:
            print(f"⚠ Warning: Maximum impact value is very large: {max_impact:.2f}")
        checks_passed = False
    else:
        if verbose:
            print(f"✓ Impact values are reasonable (max: {max_impact:.4f})")
    
    # Check 4: Treatment variable should have high impact
    treatment_var = get_treatment_variable()
    if treatment_var in results_df['variable'].values:
        treatment_row = results_df[results_df['variable'] == treatment_var].iloc[0]
        treatment_avg = treatment_row[method_cols].mean()
        overall_median = results_df[method_cols].median().median()
        
        if treatment_avg > overall_median:
            if verbose:
                print(f"✓ Treatment variable has above-median impact ({treatment_avg:.4f} > {overall_median:.4f})")
        else:
            if verbose:
                print(f"⚠ Warning: Treatment variable has below-median impact")
            checks_passed = False
    
    if verbose:
        print("=" * 80)
        print()
    
    return checks_passed


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 1A")
    print("=" * 80)
    print()
    
    # Run experiment
    results_df = run_experiment_1a(verbose=True)
    
    # Validate results
    validation_passed = validate_results(results_df, verbose=True)
    
    # Save results
    save_results(results_df)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1A COMPLETED")
    print("=" * 80)
    
    if validation_passed:
        print("✓ All validation checks passed!")
    else:
        print("⚠ Some validation checks failed. Please review results.")
