"""
Experiment 5A: Determine Distribution Types

This script tests normality and classifies variables by distribution type
and sample size.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_selected_variables(exp4_results_path: str) -> List[str]:
    """
    Load selected variables from Experiment 4.
    
    Parameters
    ----------
    exp4_results_path : str
        Path to Experiment 4 results
        
    Returns
    -------
    list
        List of selected variable names
    """
    df = pd.read_csv(exp4_results_path)
    variables = df['variable'].tolist()
    
    print(f"✓ Loaded {len(variables)} selected variables from Experiment 4")
    print(f"  Variables: {variables}")
    
    return variables


def load_ihdp_data() -> pd.DataFrame:
    """Load IHDP dataset."""
    from experiment_1.utils.data_loader import load_ihdp_data
    return load_ihdp_data()


def test_normality(data: np.ndarray, var_name: str, alpha: float = 0.05) -> Dict:
    """
    Test normality using Shapiro-Wilk test.
    
    Parameters
    ----------
    data : np.ndarray
        Variable data
    var_name : str
        Variable name
    alpha : float
        Significance level
        
    Returns
    -------
    dict
        Test results
    """
    # Shapiro-Wilk test
    statistic, p_value = stats.shapiro(data)
    
    is_normal = p_value > alpha
    
    result = {
        'variable': var_name,
        'test': 'Shapiro-Wilk',
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': is_normal,
        'alpha': alpha
    }
    
    return result


def classify_distribution(data: np.ndarray, var_name: str, 
                         small_sample_threshold: int = 30) -> Dict:
    """
    Classify variable by distribution type and sample size.
    
    Parameters
    ----------
    data : np.ndarray
        Variable data
    var_name : str
        Variable name
    small_sample_threshold : int
        Threshold for small vs large sample
        
    Returns
    -------
    dict
        Classification results
    """
    n = len(data)
    
    # Test normality
    normality_test = test_normality(data, var_name)
    is_normal = normality_test['is_normal']
    
    # Determine sample size category
    is_small = n < small_sample_threshold
    size_category = 'Small' if is_small else 'Large'
    
    # Determine distribution category
    dist_category = 'Normal' if is_normal else 'Non-normal'
    
    # Combined classification
    classification = f"{dist_category}-{size_category}"
    
    result = {
        'variable': var_name,
        'n_samples': n,
        'is_normal': is_normal,
        'is_small_sample': is_small,
        'distribution_type': dist_category,
        'sample_size_category': size_category,
        'classification': classification,
        'shapiro_statistic': normality_test['statistic'],
        'shapiro_p_value': normality_test['p_value'],
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    
    return result


def run_experiment_5a(selected_vars: List[str] = None):
    """
    Run Experiment 5A: Determine Distribution Types
    
    Parameters
    ----------
    selected_vars : list, optional
        List of variables to test. If None, loads from Experiment 4.
    """
    print("=" * 80)
    print("EXPERIMENT 5A: DETERMINE DISTRIBUTION TYPES")
    print("=" * 80)
    print()
    
    # Load selected variables if not provided
    if selected_vars is None:
        exp4_results = Path(__file__).parent.parent / 'experiment_4' / 'results' / 'selection_avg_impact_75pct.csv'
        selected_vars = load_selected_variables(exp4_results)
    
    print(f"\nTesting {len(selected_vars)} variables...")
    print()
    
    # Load IHDP data
    data = load_ihdp_data()
    
    # Test each variable
    results = []
    
    for var in selected_vars:
        if var not in data.columns:
            print(f"⚠ Warning: {var} not in dataset, skipping")
            continue
        
        var_data = data[var].values
        
        # Classify distribution
        classification = classify_distribution(var_data, var)
        results.append(classification)
        
        print(f"  {var}:")
        print(f"    Classification: {classification['classification']}")
        print(f"    Shapiro-Wilk p-value: {classification['shapiro_p_value']:.4f}")
        print(f"    Sample size: {classification['n_samples']}")
        print(f"    Mean ± SD: {classification['mean']:.3f} ± {classification['std']:.3f}")
        print()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary
    print("=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    
    classification_counts = results_df['classification'].value_counts()
    print("\nDistribution by classification:")
    for classification, count in classification_counts.items():
        print(f"  {classification}: {count} variables")
    
    # Save results
    output_path = Path(__file__).parent / 'results'
    output_path.mkdir(exist_ok=True, parents=True)
    
    csv_path = output_path / 'distribution_classifications.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results: {csv_path}")
    
    json_path = output_path / 'distribution_classifications.json'
    # Convert numpy types to Python types for JSON serialization
    results_json = []
    for r in results:
        r_copy = r.copy()
        for key, val in r_copy.items():
            if isinstance(val, (np.bool_, np.integer, np.floating)):
                r_copy[key] = val.item()
        results_json.append(r_copy)
    
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Saved JSON: {json_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 5A COMPLETED")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 5A")
    print("=" * 80)
    print()
    
    # Run experiment
    results_df = run_experiment_5a()
    
    print("\n✓ Distribution classification completed successfully!")
