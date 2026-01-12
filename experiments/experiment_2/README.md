# Experiment 2: Quadrant Division by Impact Value

## Overview

This directory contains **Experiment 2** which divides variables into quadrants based on **cumulative impact value** (not count). Each quadrant contains variables that cumulatively contribute to 25% of the total impact.

## Key Concept

Unlike traditional quartiles that divide by **count** (each quartile has 25% of variables), this experiment divides by **value**:

- **Q1**: Top variables contributing to first 25% of total impact
- **Q2**: Next variables contributing to next 25% of total impact
- **Q3**: Next variables contributing to next 25% of total impact
- **Q4**: Remaining variables (last 25% of total impact)

**Result**: Quadrants may have different numbers of variables!

## Experiments

### Experiment 2A: Value-Based Quadrants

Divides variables into quadrants based on cumulative impact.

**Algorithm**:

1. Load impact values from Experiment 1A
2. Calculate `total_impact = sum(all impact values)`
3. Sort variables by impact (descending)
4. Assign to quadrants cumulatively:
   - Q1: variables until cumulative ≥ 25% of total
   - Q2: variables until cumulative ≥ 50% of total
   - Q3: variables until cumulative ≥ 75% of total
   - Q4: remaining variables

**Run**:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_2
python run_experiment_2a.py
```

**Output**:

- `results/quadrants_<method>.json` - Quadrant assignments
- `results/quadrants_detailed_<method>.csv` - Detailed listing
- `results/quadrants_summary_<method>.csv` - Summary statistics

### Experiment 2B: Validate Quadrant Distribution

Validates that division correctly allocates ~25% of impact to each quadrant.

**Validation Checks**:

- ✓ sum(Q1 impacts) ≈ 25% of total
- ✓ sum(Q2 impacts) ≈ 25% of total
- ✓ sum(Q3 impacts) ≈ 25% of total
- ✓ sum(Q4 impacts) ≈ 25% of total
- ✓ Total = 100%

**Run**:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_2
python run_experiment_2b.py
```

**Output**:

- Console validation report
- `results/figures/quadrant_distribution_<method>.png` - 4-panel visualization
- `results/figures/quadrant_comparison_all_methods.png` - Method comparison

## Example

With 10 variables and total impact = 100:

| Rank | Variable  | Impact   | Cumulative | Quadrant |
| ---- | --------- | -------- | ---------- | -------- |
| 1    | treatment | 30       | 30%        | Q1       |
| 2    | mu1       | 20       | 50%        | Q2       |
| 3    | x16       | 15       | 65%        | Q2       |
| 4    | x6        | 10       | 75%        | Q3       |
| 5-10 | others    | 25 total | 100%       | Q4       |

**Result**: Q1=1 var, Q2=2 vars, Q3=1 var, Q4=5 vars

## Quick Start

Run both experiments sequentially:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_2

# Step 1: Divide into quadrants
python run_experiment_2a.py

# Step 2: Validate and visualize
python run_experiment_2b.py
```

## Visualizations

### Experiment 2B generates:

1. **Impact Distribution Bar Chart** - Total impact per quadrant
2. **Variable Count Bar Chart** - Number of variables per quadrant
3. **Cumulative Impact Curve** - Shows how impact accumulates
4. **Pie Chart** - Impact percentage distribution
5. **Method Comparison** - Compare quadrant divisions across methods

## Dependencies

Uses same dependencies as Experiment 1:

```bash
pip install pandas numpy matplotlib seaborn
```

## Results Interpretation

### Variable Counts

**High-impact concentration**: If Q1 has few variables, impact is concentrated in top vars  
**Even distribution**: If quadrants have similar counts, impact is spread evenly

### Impact Distribution

Should be **~25% per quadrant** (within tolerance of ±2%)

### Method Consistency

Compare across methods (linear regression, partial dependence, etc.) to see if:

- Same variables appear in Q1 (high agreement)
- Different quadrant assignments (method-dependent)

## Files Structure

```
experiment_2/
├── run_experiment_2a.py        # Main script for 2A
├── run_experiment_2b.py        # Main script for 2B
├── results/
│   ├── quadrants_*.json        # Quadrant assignments
│   ├── quadrants_detailed_*.csv
│   ├── quadrants_summary_*.csv
│   └── figures/                # Visualizations
└── README.md                   # This file
```

## Notes

- Requires Experiment 1A results as input
- Processes all 5 methods from Experiment 1A by default
- Tolerance for validation: ±2 percentage points from 25%
- All visualizations saved as high-res PNG (300 DPI)

## Troubleshooting

**Error: "No such file"**

- Run Experiment 1A first: `cd ../experiment_1 && python run_experiment_1a.py`

**Warning: "Exceeds tolerance"**

- Expected behavior if impact is highly skewed
- Check if a few variables dominate total impact

## Citation

Based on cumulative distribution analysis from impact value calculations in Experiment 1.
