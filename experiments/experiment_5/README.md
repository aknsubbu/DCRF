# Experiment 5: Statistical Test Application

## Overview

This directory contains **Experiment 5** which applies appropriate statistical tests to establish variance ranges and significance for selected variables from Experiment 4.

## Experiments

### Experiment 5A: Determine Distribution Types

Classifies each selected variable by distribution type and sample size.

**Classification Process**:

1. **Normality Test**: Shapiro-Wilk test (α = 0.05)
2. **Sample Size**: Small (n < 30) vs Large (n ≥ 30)
3. **Classification**: Combination of both

**Categories**:

- `Normal-Small`: Normal distribution, n < 30 → Use **t-test**
- `Normal-Large`: Normal distribution, n ≥ 30 → Use **Z-test**
- `Non-normal-Small`: Non-normal, n < 30 → Use **Mann-Whitney U**
- `Non-normal-Large`: Non-normal, n ≥ 30 → Use **Wilcoxon**

**Run**:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_5
python run_experiment_5a.py
```

**Output**:

- `results/distribution_classifications.csv`
- `results/distribution_classifications.json`

### Experiment 5B: Apply Statistical Tests

Applies appropriate test based on distribution classification.

**Testing Strategy**:

1. Split data by variable median:
   - Group 1: observations ≤ median
   - Group 2: observations > median
2. Compare outcome distributions between groups
3. Apply test based on classification (from 5A)
4. Calculate effect sizes

**Tests Applied**:

- **T-test**: Independent samples t-test + Cohen's d
- **Z-test**: Normal approximation + Cohen's d
- **Mann-Whitney U**: Nonparametric + Rank-biserial correlation
- **Wilcoxon**: Signed-rank (or Mann-Whitney for independent)

**Run**:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_5
python run_experiment_5b.py
```

**Output**:

- `results/statistical_tests.csv`
- `results/statistical_tests.json`

### Experiment 5C: Variance Contribution Analysis

Alternative approach using model comparison.

**Method**:

1. Fit full model: `Y ~ X₁ + X₂ + ... + Xₙ`
2. For each variable Xᵢ:
   - Fit reduced model: `Y ~ X₁ + ... + Xᵢ₋₁ + Xᵢ₊₁ + ... + Xₙ`
   - Calculate: `ΔR² = R²_full - R²_reduced`
   - Test significance using F-test
3. Rank variables by variance contribution

**Run**:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_5
python run_experiment_5c.py
```

**Output**:

- `results/variance_contributions.csv`
- `results/variance_contributions.json`

## Quick Start

Run all three experiments:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_5

# Step 1: Classify distributions
python run_experiment_5a.py

# Step 2: Apply statistical tests
python run_experiment_5b.py

# Step 3: Variance contribution analysis
python run_experiment_5c.py
```

## Interpretation

### Experiment 5A Results

**Normal-Large**: Most IHDP variables (n=747 > 30)

- Use Z-test for comparisons
- Parametric methods appropriate

**Non-normal-Large**: Some skewed variables

- Use Mann-Whitney U or Wilcoxon
- Nonparametric methods safer

### Experiment 5B Results

**Significant p-value (< 0.05)**:

- Variable creates meaningful outcome difference
- Important for stratification/subgroup analysis

**Large effect size**:

- Practical significance
- Strong predictor of outcome

### Experiment 5C Results

**High variance contribution**:

- Variable explains substantial outcome variance
- Important for predictive models
- Key for understanding outcome drivers

**Significant F-test**:

- Contribution statistically reliable
- Not due to chance

## Use Cases

### Model Selection

Use variance contribution to:

- Rank predictors by importance
- Select parsimonious model
- Identify redundant variables

### Hypothesis Testing

Use statistical tests to:

- Confirm variable-outcome relationships
- Establish significance thresholds
- Support causal claims

### Stratification

Use significant variables to:

- Define patient subgroups
- Personalize treatments
- Target interventions

## Dependencies

Requires:

- Experiment 4 results (selected variables)
- Experiment 1 data loader utilities
- `scipy`, `statsmodels`, `scikit-learn`

## Files Structure

```
experiment_5/
├── run_experiment_5a.py        # Distribution classification
├── run_experiment_5b.py        # Statistical tests
├── run_experiment_5c.py        # Variance contribution
├── results/
│   ├── distribution_classifications.csv
│   ├── statistical_tests.csv
│   └── variance_contributions.csv
└── README.md                   # This file
```

## Statistical Tests Reference

| Classification   | Test           | Effect Size   | Use Case             |
| ---------------- | -------------- | ------------- | -------------------- |
| Normal-Small     | T-test         | Cohen's d     | Small normal samples |
| Normal-Large     | Z-test         | Cohen's d     | Large normal samples |
| Non-normal-Small | Mann-Whitney U | Rank-biserial | Small skewed data    |
| Non-normal-Large | Wilcoxon       | Rank-biserial | Large skewed data    |

## Effect Size Interpretation

**Cohen's d**:

- Small: 0.2
- Medium: 0.5
- Large: 0.8

**Rank-biserial correlation**:

- Range: -1 to +1
- Interpretation similar to correlation

## Notes

- All tests are **two-sided** (testing for any difference)
- Significance level: α = 0.05
- Group splitting by **median** ensures balanced groups
- Variance contribution uses **linear regression** (R²)

## Troubleshooting

**Error: "FileNotFoundError" in 5B/5C**

- Run Experiment 5A first

**Warning: "All variables Normal-Large"**

- Expected with IHDP (n=747)
- IHDP has large sample size

**Very low R² in 5C**

- Normal if only 3 variables selected (from Exp 4 constraint)
- More variables → higher R²

## Citation

Statistical methods based on standard hypothesis testing framework and model comparison using F-tests.
