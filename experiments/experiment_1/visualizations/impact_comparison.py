"""
Visualization utilities for Experiment 1A: Impact Value Comparisons
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_impact_comparison_bar(results_df, output_path=None, top_n=15):
    """
    Create grouped bar chart comparing impact values across methods.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from Experiment 1A
    output_path : str or Path, optional
        Path to save figure
    top_n : int
        Number of top variables to show
    """
    # Calculate average impact and select top N
    results_df['avg_impact'] = results_df[[
        'linear_regression', 'partial_dependence', 
        'do_calculus', 'causal_forest'
    ]].mean(axis=1)
    
    top_vars = results_df.nlargest(top_n, 'avg_impact')
    
    # Prepare data for plotting
    plot_data = top_vars.melt(
        id_vars=['variable'],
        value_vars=['linear_regression', 'partial_dependence', 'do_calculus', 'causal_forest'],
        var_name='Method',
        value_name='Impact'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot
    sns.barplot(
        data=plot_data,
        x='variable',
        y='Impact',
        hue='Method',
        ax=ax,
        palette='Set2'
    )
    
    ax.set_xlabel('Variable', fontsize=12, fontweight='bold')
    ax.set_ylabel('Impact Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Variables by Causal Impact (Comparison Across Methods)', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.show()
    return fig


def plot_impact_heatmap(results_df, output_path=None, top_n=20):
    """
    Create heatmap of impact values (variables × methods).
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from Experiment 1A
    output_path : str or Path, optional
        Path to save figure
    top_n : int
        Number of top variables to show
    """
    # Select top N by average impact
    results_df['avg_impact'] = results_df[[
        'linear_regression', 'partial_dependence', 
        'do_calculus', 'causal_forest'
    ]].mean(axis=1)
    
    top_vars = results_df.nlargest(top_n, 'avg_impact')
    
    # Prepare matrix
    heatmap_data = top_vars.set_index('variable')[[
        'linear_regression', 'partial_dependence', 
        'do_calculus', 'causal_forest'
    ]]
    
    # Rename columns for better display
    heatmap_data.columns = ['Linear Reg', 'Partial Dep', 'Do-Calculus', 'Causal Forest']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Impact Value'},
        ax=ax
    )
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
    ax.set_title(f'Impact Value Heatmap (Top {top_n} Variables)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.show()
    return fig


def plot_method_correlation(results_df, output_path=None):
    """
    Create scatter plots comparing methods pairwise.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from Experiment 1A
    output_path : str or Path, optional
        Path to save figure
    """
    methods = ['linear_regression', 'partial_dependence', 'do_calculus', 'causal_forest']
    method_names = ['Linear Reg', 'Partial Dep', 'Do-Calculus', 'Causal Forest']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    plot_idx = 0
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            ax = axes[plot_idx]
            
            x = results_df[methods[i]]
            y = results_df[methods[j]]
            
            # Scatter plot
            ax.scatter(x, y, alpha=0.6, s=50)
            
            # Add diagonal line
            max_val = max(x.max(), y.max())
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
            
            # Calculate correlation
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(method_names[i], fontsize=10)
            ax.set_ylabel(method_names[j], fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Method Correlation Analysis', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.show()
    return fig


def plot_ranking_comparison(results_df, output_path=None, top_k=10):
    """
    Compare top-K variable rankings across methods.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from Experiment 1A
    output_path : str or Path, optional
        Path to save figure
    top_k : int
        Number of top variables to compare
    """
    methods = ['linear_regression', 'partial_dependence', 'do_calculus', 'causal_forest']
    method_names = ['Linear Reg', 'Partial Dep', 'Do-Calculus', 'Causal Forest']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, (method, method_name) in enumerate(zip(methods, method_names)):
        ax = axes[idx]
        
        # Get top K variables for this method
        top_k_vars = results_df.nlargest(top_k, method)
        
        # Plot
        colors = sns.color_palette('viridis', top_k)
        ax.barh(range(top_k), top_k_vars[method].values, color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_k_vars['variable'].values)
        ax.set_xlabel('Impact Value', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_k} Variables: {method_name}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, val in enumerate(top_k_vars[method].values):
            ax.text(val, i, f'  {val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.show()
    return fig


def create_all_visualizations(results_csv_path, output_dir='results/figures'):
    """
    Create all visualizations for Experiment 1A.
    
    Parameters
    ----------
    results_csv_path : str or Path
        Path to Experiment 1A results CSV
    output_dir : str or Path
        Directory to save figures
    """
    # Load results
    results_df = pd.read_csv(results_csv_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Creating visualizations for Experiment 1A...")
    print()
    
    # Generate plots
    print("1. Impact Comparison Bar Chart...")
    plot_impact_comparison_bar(
        results_df, 
        output_path / 'impact_comparison_bar.png'
    )
    
    print("2. Impact Heatmap...")
    plot_impact_heatmap(
        results_df,
        output_path / 'impact_heatmap.png'
    )
    
    print("3. Method Correlation...")
    plot_method_correlation(
        results_df,
        output_path / 'method_correlation.png'
    )
    
    print("4. Ranking Comparison...")
    plot_ranking_comparison(
        results_df,
        output_path / 'ranking_comparison.png'
    )
    
    print()
    print(f"✓ All visualizations saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Assume results CSV is in results/ directory
    results_path = Path(__file__).parent.parent / 'results' / 'experiment_1a_results.csv'
    
    if results_path.exists():
        create_all_visualizations(results_path)
    else:
        print(f"Results file not found: {results_path}")
        print("Please run run_experiment_1a.py first.")
