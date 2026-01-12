# Experiment 1: Impact Value Calculation

## Overview

This directory contains the implementation of **Experiment 1** which calculates **impact values** (causal effects) for variables in the IHDP dataset based on the FCI Graph (PAG).

## Experiments

### Experiment 1A: Direct Causal Effect Estimation

Calculates impact values using four different methods:

1. **Linear Regression** - Standardized regression coefficients
2. **Partial Dependence** - Slope of PDP curves
3. **Do-Calculus** - Backdoor adjustment estimation
4. **Causal Forest** - Variable importance (RF proxy)

**Run:**

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_1
python run_experiment_1a.py
```

**Output:**

- `results/experiment_1a_results.csv` - Impact values for all variables
- Console output with top variables and validation

### Experiment 1B: Impact Value Ranges

Extends 1A by calculating confidence intervals using:

1. **Bootstrap Confidence Intervals** (1000 iterations)
2. **Analytical Confidence Intervals** (when available)

Provides conservative estimates using lower bounds.

**Run:**

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_1
python run_experiment_1b.py
```

**Output:**

- `results/experiment_1b_results.csv` - Results with CIs
- Comparison of bootstrap vs analytical methods

## Project Structure

```
experiment_1/
├── utils/
│   ├── data_loader.py          # Load IHDP data and FCI PAG
│   ├── pag_analysis.py          # PAG analysis (adjustment sets)
│   └── effect_estimators.py    # Causal effect estimators
├── visualizations/
│   ├── impact_comparison.py    # Visualizations for Experiment 1A
│   └── confidence_intervals.py # Visualizations for Experiment 1B
├── results/
│   ├── experiment_1a_results.csv
│   ├── experiment_1b_results.csv
│   └── figures/                # Generated plots
├── tests/
│   └── (test files)
├── run_experiment_1a.py        # Main script for Experiment 1A
├── run_experiment_1b.py        # Main script for Experiment 1B
└── README.md                   # This file
```

## Installation

Required packages (should already be installed):

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

## Quick Start

Run both experiments sequentially:

```bash
# Navigate to experiment directory
cd /Volumes/DevDrive/DCRF/experiments/experiment_1

# Run Experiment 1A
python run_experiment_1a.py

# Run Experiment 1B
python run_experiment_1b.py

# Generate visualizations
python visualizations/impact_comparison.py
```

## Results Interpretation

### Impact Values

- **Higher values** = Stronger causal effect on outcome (`y_factual`)
- Values are **absolute** (always non-negative)
- Treatment variable should typically have high impact

### Methods Comparison

- **Linear Regression**: Fast, interpretable, assumes linearity
- **Partial Dependence**: Flexible, captures non-linear effects
- **Do-Calculus**: Theory-grounded, adjusts for confounders
- **Causal Forest**: Data-driven, variable importance

### Confidence Intervals

- **Bootstrap**: Distribution-free, robust
- **Analytical**: Faster, requires assumptions
- **Conservative Estimate**: Lower bound for safety

## Validation

Each experiment includes built-in validation:

- ✓ No NaN values
- ✓ Non-negative impacts
- ✓ Reasonable magnitudes
- ✓ Treatment has high impact
- ✓ Valid confidence intervals

## Troubleshooting

**Error: "FileNotFoundError: IHDP data not found"**

- Ensure `data/ihdp.csv` exists in project root

**Error: "FCI PAG not found"**

- Ensure `fci_adjacency_matrix.csv` exists in project root
- Run `fci.ipynb` to generate it

**Warning: "Some validation checks failed"**

- Review console output for specific issues
- Check data quality and PAG structure

## Notes

- Experiment 1B uses 100 bootstrap iterations by default for speed
- For final results, increase to 1000 iterations (edit `run_experiment_1b.py`)
- All random operations use seed=42 for reproducibility

## Citation

Based on:

- IHDP dataset
- FCI (Fast Causal Inference) algorithm
- Causal effect estimation methods from causal inference literature
