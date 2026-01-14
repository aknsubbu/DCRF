"""
Analyze the Distribution of Impact Values and Test for Normality

This script analyzes the distribution of impact values from Experiment 1A
and tests whether they follow a normal distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_results():
    """Load experiment 1A results."""
    results_path = Path(__file__).parent / 'results' / 'experiment_1a_results.csv'
    return pd.read_csv(results_path)


def test_normality(data, var_name):
    """
    Perform multiple normality tests on the data.
    
    Parameters
    ----------
    data : array-like
        Data to test
    var_name : str
        Name of the variable for reporting
        
    Returns
    -------
    dict
        Dictionary with test results
    """
    results = {'variable': var_name}
    
    # Remove NaN values
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    if len(data) < 3:
        return None
    
    # 1. Shapiro-Wilk Test (best for small samples n < 50)
    stat, p_value = stats.shapiro(data)
    results['shapiro_statistic'] = stat
    results['shapiro_pvalue'] = p_value
    results['shapiro_normal'] = p_value > 0.05
    
    # 2. D'Agostino-Pearson Test (for n >= 20)
    if len(data) >= 20:
        stat, p_value = stats.normaltest(data)
        results['dagostino_statistic'] = stat
        results['dagostino_pvalue'] = p_value
        results['dagostino_normal'] = p_value > 0.05
    else:
        results['dagostino_statistic'] = np.nan
        results['dagostino_pvalue'] = np.nan
        results['dagostino_normal'] = np.nan
    
    # 3. Anderson-Darling Test
    anderson_result = stats.anderson(data, dist='norm')
    results['anderson_statistic'] = anderson_result.statistic
    # Compare with critical value at 5% significance
    results['anderson_critical_5pct'] = anderson_result.critical_values[2]  # 5% level
    results['anderson_normal'] = anderson_result.statistic < anderson_result.critical_values[2]
    
    # 4. Kolmogorov-Smirnov Test
    # Standardize data for K-S test against standard normal
    standardized_data = (data - np.mean(data)) / np.std(data)
    stat, p_value = stats.kstest(standardized_data, 'norm')
    results['ks_statistic'] = stat
    results['ks_pvalue'] = p_value
    results['ks_normal'] = p_value > 0.05
    
    # Descriptive statistics
    results['mean'] = np.mean(data)
    results['std'] = np.std(data)
    results['skewness'] = stats.skew(data)
    results['kurtosis'] = stats.kurtosis(data)
    results['n_samples'] = len(data)
    
    return results


def plot_distribution(data, title, ax):
    """Plot histogram with normal distribution overlay."""
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    # Histogram
    ax.hist(data, bins='auto', density=True, alpha=0.7, color='steelblue', edgecolor='white')
    
    # Fit normal distribution
    mu, std = stats.norm.fit(data)
    x = np.linspace(data.min() - 0.5*std, data.max() + 0.5*std, 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal fit (μ={mu:.3f}, σ={std:.3f})')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Impact Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)


def plot_qq(data, title, ax):
    """Create Q-Q plot."""
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot: {title}', fontsize=12, fontweight='bold')


def analyze_advantages_of_normality():
    """
    Explain the advantages of normally distributed impact values.
    
    Returns
    -------
    str
        Explanation text
    """
    advantages = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           ADVANTAGES OF NORMALLY DISTRIBUTED IMPACT VALUES                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. STATISTICAL INFERENCE
   ─────────────────────
   • Confidence intervals can be calculated using well-established formulas
     (e.g., x̄ ± 1.96σ for 95% CI)
   • Hypothesis testing (t-tests, ANOVA) can be applied directly
   • Standard errors are meaningful and interpretable

2. PREDICTIVE MODELING
   ────────────────────
   • Linear regression assumptions are satisfied
   • Maximum likelihood estimation is optimal
   • Model residuals are well-behaved

3. ROBUST COMPARISONS
   ───────────────────
   • Z-scores allow standardized comparison across variables
   • Ranking and percentile calculations are reliable
   • Effect sizes (Cohen's d) are meaningful

4. INTERPRETABILITY
   ─────────────────
   • The mean is the best measure of central tendency
   • 68-95-99.7 rule applies (% within 1, 2, 3 std deviations)
   • Outliers can be objectively identified (> 3σ)

5. AGGREGATION PROPERTIES
   ───────────────────────
   • Sums and averages of normal variables remain normal
   • Central Limit Theorem ensures sample means converge to normal
   • Portfolio-like combinations of impacts are predictable

6. CAUSAL INFERENCE IMPLICATIONS
   ──────────────────────────────
   • Additive causal effects are easier to estimate and interpret
   • Backdoor adjustment and inverse probability weighting work better
   • Sensitivity analyses have well-defined bounds

⚠️  IF NOT NORMALLY DISTRIBUTED:
   ──────────────────────────────
   • May need non-parametric methods (median, IQR instead of mean, std)
   • Bootstrap confidence intervals become necessary
   • Transformations (log, Box-Cox) may be required
   • Heavy tails suggest a few variables dominate (Pareto-like)
"""
    return advantages


def run_distribution_analysis():
    """Run complete distribution analysis."""
    print("=" * 80)
    print("IMPACT VALUE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()
    
    # Load results
    df = load_results()
    
    methods = ['linear_regression', 'partial_dependence', 'do_calculus', 'causal_forest', 'avg_impact']
    
    # Perform normality tests
    print("STEP 1: NORMALITY TESTS")
    print("-" * 80)
    
    all_results = []
    for method in methods:
        result = test_normality(df[method], method)
        if result:
            all_results.append(result)
            
            print(f"\n📊 {method.upper().replace('_', ' ')}")
            print(f"   Sample size: {result['n_samples']}")
            print(f"   Mean: {result['mean']:.4f}, Std: {result['std']:.4f}")
            print(f"   Skewness: {result['skewness']:.4f}, Kurtosis: {result['kurtosis']:.4f}")
            print()
            print(f"   Shapiro-Wilk Test:")
            print(f"     Statistic: {result['shapiro_statistic']:.4f}, p-value: {result['shapiro_pvalue']:.4f}")
            print(f"     Normal? {'✓ YES' if result['shapiro_normal'] else '✗ NO'} (p > 0.05)")
            print()
            print(f"   Anderson-Darling Test:")
            print(f"     Statistic: {result['anderson_statistic']:.4f}, Critical (5%): {result['anderson_critical_5pct']:.4f}")
            print(f"     Normal? {'✓ YES' if result['anderson_normal'] else '✗ NO'} (stat < critical)")
            print()
            print(f"   Kolmogorov-Smirnov Test:")
            print(f"     Statistic: {result['ks_statistic']:.4f}, p-value: {result['ks_pvalue']:.4f}")
            print(f"     Normal? {'✓ YES' if result['ks_normal'] else '✗ NO'} (p > 0.05)")
    
    results_df = pd.DataFrame(all_results)
    
    # Summary table
    print("\n" + "=" * 80)
    print("STEP 2: NORMALITY TEST SUMMARY")
    print("-" * 80)
    
    summary = results_df[['variable', 'shapiro_normal', 'anderson_normal', 'ks_normal', 'skewness', 'kurtosis']].copy()
    summary.columns = ['Method', 'Shapiro (Normal?)', 'Anderson (Normal?)', 'K-S (Normal?)', 'Skewness', 'Kurtosis']
    print(summary.to_string(index=False))
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Impact Value Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    for i, method in enumerate(methods):
        # Distribution plot
        plot_distribution(df[method], method.replace('_', ' ').title(), axes[0, i])
        # Q-Q plot
        plot_qq(df[method], method.replace('_', ' ').title(), axes[1, i])
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / 'results' / 'figures'
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / 'impact_distribution_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Distribution plots saved to: {fig_path}")
    
    # Box plot comparison
    fig2, ax = plt.subplots(figsize=(12, 6))
    df_melted = df.melt(id_vars=['variable'], value_vars=['linear_regression', 'partial_dependence', 'do_calculus', 'causal_forest'],
                        var_name='Method', value_name='Impact')
    sns.boxplot(x='Method', y='Impact', data=df_melted, ax=ax, palette='husl')
    ax.set_title('Impact Value Distribution by Method', fontsize=14, fontweight='bold')
    ax.set_xlabel('Estimation Method', fontsize=12)
    ax.set_ylabel('Impact Value', fontsize=12)
    ax.set_xticklabels([m.replace('_', '\n') for m in ['linear_regression', 'partial_dependence', 'do_calculus', 'causal_forest']])
    
    fig2_path = output_dir / 'impact_boxplot_comparison.png'
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Box plot saved to: {fig2_path}")
    
    # Print advantages
    print("\n" + "=" * 80)
    print("STEP 4: ADVANTAGES OF NORMAL DISTRIBUTION")
    print(analyze_advantages_of_normality())
    
    # Final interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION & RECOMMENDATIONS")
    print("-" * 80)
    
    normal_count = results_df['shapiro_normal'].sum()
    total_count = len(results_df)
    
    if normal_count == total_count:
        print("""
✓ All impact value distributions appear to be NORMALLY DISTRIBUTED.

This is excellent news for your causal analysis because:
1. Standard statistical methods can be applied directly
2. Confidence intervals are reliable
3. Impact values are well-behaved for aggregation and comparison
4. The mean is a robust summary statistic
""")
    elif normal_count >= total_count / 2:
        print(f"""
⚡ {normal_count}/{total_count} distributions appear to be normally distributed.

This suggests:
1. Some methods produce more normally distributed impacts than others
2. Consider using methods with normal distributions for primary analysis
3. For non-normal distributions, consider:
   - Log transformation if right-skewed
   - Bootstrap confidence intervals
   - Non-parametric comparisons (Mann-Whitney U, Kruskal-Wallis)
""")
    else:
        print(f"""
⚠️ Only {normal_count}/{total_count} distributions appear normally distributed.

Recommendations:
1. Use MEDIAN and IQR instead of mean and std for central tendency
2. Apply Bootstrap methods for confidence intervals
3. Consider transformations:
   - Log transform for right-skewed (many small values, few large)
   - Box-Cox transformation for general normalization
4. Use non-parametric statistical tests
5. For ranking, use percentiles rather than z-scores
""")
    
    # Check for skewness issues
    right_skewed = (results_df['skewness'] > 1).sum()
    left_skewed = (results_df['skewness'] < -1).sum()
    heavy_tailed = (results_df['kurtosis'] > 3).sum()
    
    if right_skewed > 0:
        print(f"\n📈 {right_skewed} distribution(s) are RIGHT-SKEWED (skewness > 1)")
        print("   This suggests a few variables have very high impact values.")
        print("   Consider log transformation: log(impact + 1)")
    
    if left_skewed > 0:
        print(f"\n📉 {left_skewed} distribution(s) are LEFT-SKEWED (skewness < -1)")
        print("   This suggests most variables cluster at high values.")
    
    if heavy_tailed > 0:
        print(f"\n🔔 {heavy_tailed} distribution(s) are HEAVY-TAILED (kurtosis > 3)")
        print("   This suggests more extreme values than a normal distribution.")
        print("   May indicate heterogeneous effect magnitudes.")
    
    print("\n" + "=" * 80)
    
    # Save results
    results_csv_path = output_dir.parent / 'normality_test_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n✓ Normality test results saved to: {results_csv_path}")
    
    plt.show()
    
    return results_df


if __name__ == "__main__":
    run_distribution_analysis()
