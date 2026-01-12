# Experiment 3: Variable Classification by Causal Role

## Overview

This directory contains **Experiment 3** which classifies variables by their **causal role** in the PAG (Partial Ancestral Graph) and cross-references with quadrant assignments from Experiment 2.

## Variable Roles

### Independent Variables

- Variables on any **directed path** from treatment → outcome
- These are mediators or variables through which treatment affects outcome
- Key for understanding causal mechanisms

### Confounders

- Variables with paths to **BOTH** treatment AND outcome
- Must be controlled for unbiased treatment effect estimation
- Critical for causal inference

### Edge Variables

- Variables with edges in PAG but **NOT** on treatment→outcome path
- Connected to the causal system but not in main pathway
- May interact with other variables

### Other

- Variables not falling into above categories
- Isolated or weakly connected
- May be noise or irrelevant

## Experiments

### Experiment 3A: Extract Variable Roles from PAG

Classifies all variables by analyzing PAG structure.

**Algorithm**:

1. Identify directed paths from treatment → outcome
2. Variables on these paths = **Independent**
3. Variables with paths to both treatment AND outcome = **Confounder**
4. Variables with edges but not in above categories = **Edge**
5. Remaining variables = **Other**

**Run**:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_3
python run_experiment_3a.py
```

**Output**:

- `results/variable_roles.json` - Role assignments
- `results/variable_roles.csv` - CSV format
- `results/variables_by_role.json` - Grouped by role

### Experiment 3B: Cross-Reference with Quadrants

Creates matrix showing distribution of variable types across impact quadrants.

**Matrix Format**:

```
|    | Independent | Confounder | Edge | Other |
|----|-------------|------------|------|-------|
| Q1 | var1, var2  | var5       | var3 |       |
| Q2 | var6        | var7, var8 | var4 | var9  |
| Q3 |             | var10      | ...  | ...   |
| Q4 |             |            | ...  | ...   |
```

**Run**:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_3
python run_experiment_3b.py
```

**Output**:

- `results/crosstab_counts_<method>.csv` - Count matrix
- `results/crosstab_detailed_<method>.csv` - With variable names
- `results/crosstab_<method>.json` - Complete data
- `results/figures/crosstab_roles_quadrants_<method>.png` - Heatmap
- `results/figures/role_distribution.png` - Overall distribution

## Quick Start

Run both experiments:

```bash
cd /Volumes/DevDrive/DCRF/experiments/experiment_3

# Step 1: Classify variables by role
python run_experiment_3a.py

# Step 2: Cross-reference with quadrants
python run_experiment_3b.py
```

## Interpretation

### High-Impact Independent Variables (Q1/Q2)

- Top-priority mediators
- Understanding mechanism
- Intervention targets

### High-Impact Confounders (Q1/Q2)

- Must control in analysis
- Risk of bias if ignored
- Important for valid causal estimates

### Low-Impact Variables (Q3/Q4)

- May be less critical
- Could be excluded in sensitivity analysis
- Useful for robustness checks

### Insights from Cross-Tab

**Many confounders in Q1/Q2**:

- Complex confounding structure
- Need careful adjustment

**Many independent vars in Q1**:

- Strong mediation pathway
- Treatment works through these variables

**Edges concentrated in Q4**:

- Peripheral variables
- Less critical for main causal question

## Dependencies

Uses utilities from Experiment 1:

```python
from experiment_1.utils.pag_analysis import get_causal_paths, find_parents
```

Requires results from Experiment 2 for cross-tabulation.

## Files Structure

```
experiment_3/
├── run_experiment_3a.py        # Role classification
├── run_experiment_3b.py        # Cross-reference with quadrants
├── results/
│   ├── variable_roles.json
│   ├── variable_roles.csv
│   ├── variables_by_role.json
│   ├── crosstab_*.csv
│   ├── crosstab_*.json
│   └── figures/               # Visualizations
└── README.md                  # This file
```

## Notes

- Requires PAG from `fci.ipynb` and `fci_adjacency_matrix.csv`
- Requires Experiment 2 results for cross-tabulation
- Treatment and outcome variables have special roles
- Path finding limited to length ≤ 10 for performance

## Troubleshooting

**Error: "No module named 'experiment_1'"**

- Script adds parent directories to path automatically
- Ensure experiment_1 folder exists

**Error: "FileNotFoundError: variable_roles.json"**

- Run Experiment 3A first before 3B

**Empty quadrants in cross-tab**

- Normal if impact is concentrated
- Some role-quadrant combinations may be empty

## Citation

Based on causal graph analysis using FCI algorithm and PAG interpretation.
