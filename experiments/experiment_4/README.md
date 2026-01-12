# Experiment 4: 75% Selection Rule

## Overview

This directory contains **Experiment 4** which implements a smart variable selection strategy: selecting **equal numbers** of variables from Q1, Q2, Q3 (excluding Q4) to cover **75% of total impact**.

## Key Concept

Traditional variable selection often takes all high-impact variables, leading to imbalanced selection. This experiment enforces **equal representation** from Q1, Q2, Q3 while achieving a specific impact coverage threshold.

**Goal**: Find the minimum equal count _n_ such that:

- Select top _n_ variables from Q1
- Select top _n_ variables from Q2
- Select top _n_ variables from Q3
- Cumulative impact ≥ 75% of total

**Q4 is excluded** entirely (lowest-impact variables).

## Experiments

### Experiment 4A: Equal Count Selection (75% Rule)

Implements the selection algorithm to find optimal equal count.

**Algorithm**:

1. Load quadrant assignments from Experiment 2
2. Exclude Q4 entirely
3. Calculate target = 0.75 × total_impact
4. For each count _n_ = 1, 2, 3, ...:
   - Select top _n_ from Q1, Q2, Q3
   - Calculate cumulative impact
   - Check if ≥ target
5. Return smallest _n_ that achieves target

**Run**:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_4
python run_experiment_4a.py
```

**Output**:

- `results/selection_<method>_75pct.json` - Selection details
- `results/selection_<method>_75pct.csv` - Selected variables list

### Experiment 4B: Sensitivity Analysis

Tests whether 75% is optimal by trying multiple thresholds.

**Thresholds Tested**: 60%, 70%, 75%, 80%, 90%

For each threshold, records:

- Number of variables selected
- Total impact covered
- Count per quadrant
- Accuracy (how close to target)

**Run**:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_4
python run_experiment_4b.py
```

**Output**:

- `results/sensitivity_analysis_<method>.csv` - Full results
- `results/figures/sensitivity_analysis_<method>.png` - 4-panel plot

## Why 75%?

**Trade-off**:

- **Higher %** → More variables, more complex models
- **Lower %** → Fewer variables, may miss important effects

**75% balances**:

- Comprehensiveness (captures most impact)
- Parsimony (avoids redundant variables)
- Practical feasibility (moderate variable count)

Experiment 4B validates this choice empirically.

## Example

With avg_impact method:

- Q1 (1 var): treatment (impact = 3.38)
- Q2 (5 vars): mu1, x16, x6, y_cfactual, x15
- Q3 (8 vars): x18, mu0, x9, x22, x10, x24, x25, x17
- Q4 (15 vars): EXCLUDED

**Selection process**:

- Try n=1: Q1[1] + Q2[1] + Q3[1] = 3 vars → Check impact
- Try n=2: Q1[1] + Q2[2] + Q3[2] = 5 vars → Check impact
- ...
- Find n that achieves ≥75% impact

## Quick Start

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_4

# Run 75% selection
python run_experiment_4a.py

# Run sensitivity analysis
python run_experiment_4b.py
```

## Interpretation

### Selected Variables

Variables chosen by this method are:

- **High-impact** (from Q1-Q3, not Q4)
- **Balanced across quadrants** (equal count from each)
- **Sufficient for 75% coverage** (captures most important effects)

### Use Cases

**Model Building**:

- Use selected variables as core feature set
- Reduces overfitting vs using all variables
- Maintains interpretability

**Experimental Design**:

- Focus data collection on selected variables
- Resource allocation priority
- Cost-effective measurement

**Causal Analysis**:

- Adjustment set for treatment effect
- Sensitivity analysis variables
- Robustness checks

## Dependencies

Requires results from:

- Experiment 1 (impact values)
- Experiment 2 (quadrant assignments)

## Files Structure

```
experiment_4/
├── run_experiment_4a.py        # 75% selection
├── run_experiment_4b.py        # Sensitivity analysis
├── results/
│   ├── selection_*.json
│   ├── selection_*.csv
│   ├── sensitivity_analysis_*.csv
│   └── figures/                # Visualizations
└── README.md                   # This file
```

## Visualizations (Experiment 4B)

1. **Variables Selected vs Threshold** - Shows how count increases
2. **Achieved vs Target Coverage** - Accuracy of selection
3. **Equal Count per Quadrant** - Shows n value for each threshold
4. **Selection Precision** - Error from target

## Notes

- **Equal count constraint** may not always achieve exact 75%
- Algorithm finds **closest achievable** coverage
- Q4 exclusion is deliberate (low-impact variables)
- Works with any method from Experiment 1/2

## Troubleshooting

**Error: "Not enough variables in Q1/Q2/Q3"**

- Equal count limited by smallest quadrant
- May need to relax equal-count constraint

**Coverage far from 75%**

- Impact distribution may be very uneven
- Try different threshold in 4B
- Check quadrant sizes

**Empty quadrants**

- Run Experiment 2 first
- Ensure method parameter matches

## Citation

Selection strategy based on Pareto principle (80/20 rule) adapted for causal variable selection with balanced sampling.
