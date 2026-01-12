"""
Experiment 5B: Apply Statistical Tests

This script applies appropriate statistical tests based on distribution
classifications to compare outcome between groups.
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


def load_distribution_classifications() -> pd.DataFrame:
    """Load distribution classifications from Experiment 5A."""
    csv_path = Path(__file__).parent / 'results' / 'distribution_classifications.csv'
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded classifications for {len(df)} variables")
    return df


def load_ihdp_data() -> pd.DataFrame:
    """Load IHDP dataset."""
    from experiment_1.utils.data_loader import load_ihdp_data, get_outcome_variable
    data = load_ihdp_data()
    outcome = get_outcome_variable()
    return data, outcome


def split_by_median(data: pd.DataFrame, var: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into two groups based on variable median.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    var : str
        Variable name
        
    Returns
    -------
    tuple
        (group1 ≤ median, group2 > median)
    """
    median_val = data[var].median()
    
    group1 = data[data[var] <= median_val]
    group2 = data[data[var] > median_val]
    
    return group1, group2


def apply_t_test(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """Apply independent t-test."""
    statistic, p_value = stats.ttest_ind(group1, group2)
    
    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
    effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
    
    return {
        'test': 't-test',
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_type': "Cohen's d"
    }


def apply_z_test(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """Apply Z-test (normal approximation)."""
    from statsmodels.stats.weightstats import ztest
    
    statistic, p_value = ztest(group1, group2)
    
    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
    effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
    
    return {
        'test': 'Z-test',
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_type': "Cohen's d"
    }


def apply_mann_whitney(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """Apply Mann-Whitney U test."""
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Rank-biserial correlation as effect size
    n1, n2 = len(group1), len(group2)
    effect_size = (statistic / (n1 * n2)) * 2 - 1
    
    return {
        'test': 'Mann-Whitney U',
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_type': 'Rank-biserial'
    }


def apply_wilcoxon(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """Apply Wilcoxon signed-rank test (using Mann-Whitney for unpaired)."""
    # For independent samples, use Mann-Whitney instead
    return apply_mann_whitney(group1, group2)


def apply_appropriate_test(data: pd.DataFrame, var: str, outcome: str,
                           classification: str) -> Dict:
    """
    Apply appropriate statistical test based on classification.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    var : str
        Variable name
    outcome : str
        Outcome variable name
    classification : str
        Distribution classification
        
    Returns
    -------
    dict
        Test results
    """
    # Split into groups
    group1_data, group2_data = split_by_median(data, var)
    
    group1_outcomes = group1_data[outcome].values
    group2_outcomes = group2_data[outcome].values
    
    # Apply test based on classification
    if classification == 'Normal-Small':
        result = apply_t_test(group1_outcomes, group2_outcomes)
    elif classification == 'Normal-Large':
        result = apply_z_test(group1_outcomes, group2_outcomes)
    elif classification == 'Non-normal-Small':
        result = apply_mann_whitney(group1_outcomes, group2_outcomes)
    elif classification == 'Non-normal-Large':
        result = apply_wilcoxon(group1_outcomes, group2_outcomes)
    else:
        raise ValueError(f"Unknown classification: {classification}")
    
    # Add metadata
    result['variable'] = var
    result['classification'] = classification
    result['n_group1'] = len(group1_outcomes)
    result['n_group2'] = len(group2_outcomes)
    result['mean_group1'] = np.mean(group1_outcomes)
    result['mean_group2'] = np.mean(group2_outcomes)
    result['mean_diff'] = result['mean_group1'] - result['mean_group2']
    result['is_significant'] = result['p_value'] < 0.05
    
    return result


def run_experiment_5b():
    """
    Run Experiment 5B: Apply Statistical Tests
    """
    print("=" * 80)
    print("EXPERIMENT 5B: APPLY STATISTICAL TESTS")
    print("=" * 80)
    print()
    
    # Load classifications
    classifications_df = load_distribution_classifications()
    
    # Load data
    data, outcome = load_ihdp_data()
    
    print(f"Outcome variable: {outcome}")
    print(f"Testing {len(classifications_df)} variables...")
    print()
    
    # Apply tests
    results = []
    
    for _, row in classifications_df.iterrows():
        var = row['variable']
        classification = row['classification']
        
        print(f"  {var} ({classification}):")
        
        try:
            test_result = apply_appropriate_test(data, var, outcome, classification)
            results.append(test_result)
            
            print(f"    Test: {test_result['test']}")
            print(f"    Statistic: {test_result['statistic']:.4f}")
            print(f"    P-value: {test_result['p_value']:.4f} {'*' if test_result['is_significant'] else ''}")
            print(f"    Effect size: {test_result['effect_size']:.4f} ({test_result['effect_size_type']})")
            print(f"    Mean diff: {test_result['mean_diff']:.4f}")
            print()
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            print()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    sig_count = results_df['is_significant'].sum()
    print(f"\nSignificant results (p < 0.05): {sig_count}/{len(results_df)}")
    
    if sig_count > 0:
        print(f"\nSignificant variables:")
        sig_vars = results_df[results_df['is_significant']].sort_values('p_value')
        for _, row in sig_vars.iterrows():
            print(f"  {row['variable']}: p={row['p_value']:.4f}, effect={row['effect_size']:.3f}")
    
    # Save results
    output_path = Path(__file__).parent / 'results'
    
    csv_path = output_path / 'statistical_tests.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results: {csv_path}")
    
    json_path = output_path / 'statistical_tests.json'
    # Convert numpy types to Python types
    results_json = []
    for r in results:
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
    print("EXPERIMENT 5B COMPLETED")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 5B")
    print("=" * 80)
    print()
    
    # Run experiment
    results_df = run_experiment_5b()
    
    print("\n✓ Statistical testing completed successfully!")
