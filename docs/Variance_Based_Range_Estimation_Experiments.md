# Variance-Based Range Estimation System: Experiment Documentation

## Overview

This document outlines the experimental framework for building a **Variance-Based Range Estimation System** using causal inference techniques. The system quantifies outcome uncertainty by identifying high-impact variables through a structured pipeline that combines causal discovery, impact analysis, and statistical testing.

### Core Problem Statement

Given observational data and a causal graph (PAG from FCI algorithm), we want to:
1. Identify which variables have the greatest causal impact on the outcome
2. Select a parsimonious set of variables that explains most of the outcome variance
3. Quantify the range of possible outcome values with statistical rigor

### Why This Approach?

Traditional prediction intervals treat all variables equally. Our approach:
- **Leverages causal structure** - Uses PAG to identify true causal relationships, not mere correlations
- **Prioritizes by impact** - Focuses on variables that actually move the outcome
- **Balances coverage vs. parsimony** - The 75% rule captures most variance with fewer variables
- **Provides interpretability** - Each selected variable has a clear causal justification

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VARIANCE-BASED RANGE ESTIMATION                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [IHDP Data] + [FCI PAG]                                                │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────┐                                                    │
│  │  EXPERIMENT 1   │  Calculate impact value for each variable          │
│  │  Impact Values  │  → How much does Y change when X changes?          │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │  EXPERIMENT 2   │  Divide variables into quadrants by VALUE          │
│  │  Quadrants      │  → Q1-Q4 each contain 25% of total impact          │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │  EXPERIMENT 3   │  Classify causal roles from PAG                    │
│  │  Variable Roles │  → Independent, Confounder, Edge, Other            │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │  EXPERIMENT 4   │  Select variables covering 75% of impact           │
│  │  75% Selection  │  → Equal count from Q1, Q2, Q3 (exclude Q4)        │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │  EXPERIMENT 5   │  Apply appropriate statistical tests               │
│  │  Statistical    │  → T-test, Z-test, Mann-Whitney, Wilcoxon          │
│  │  Tests          │  → Based on distribution type + sample size        │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │  EXPERIMENT 6   │  Aggregate variance contributions                  │
│  │  Range Calc     │  → Outcome range: [lower_bound, upper_bound]       │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │  EXPERIMENT 7   │  Compare against ground truth                      │
│  │  Validation     │  → Coverage %, MAE, calibration                    │
│  └─────────────────┘                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## EXPERIMENT 1: Impact Value Calculation

### Objective
Quantify how much the outcome (Y) changes when each variable (X) changes by a standard amount.

### Why This Matters
- Not all variables are created equal - some have large effects, others negligible
- Impact values provide a principled basis for variable prioritization
- Using causal adjustment sets ensures we measure TRUE causal effects, not spurious correlations

### Experiment 1A: Direct Causal Effect Estimation

**Logic:**
The causal effect of X on Y is defined as the expected change in Y when we *intervene* on X (do-calculus). In observational data, we approximate this using:

1. **Adjustment formula**: E[Y | do(X=x)] = Σ_z E[Y | X=x, Z=z] P(Z=z)
   - Z is the adjustment set (confounders to control for)
   - PAG tells us which variables belong in Z

2. **Standardization**: We measure impact in standard deviation units to make effects comparable across variables with different scales

**Methods to Implement:**

| Method | Assumptions | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| Linear Regression | Linear relationships | Fast, interpretable coefficients | Misses non-linear effects |
| Partial Dependence | Model-agnostic | Captures non-linearity | Computationally expensive |
| Do-Calculus | Causal graph is correct | Theoretically grounded | Requires valid adjustment set |
| Causal Forest | Heterogeneous effects | Data-driven | Requires econml package |

**Input:**
- IHDP dataset (747 observations, ~25 variables)
- PAG adjacency matrix from FCI algorithm
- Outcome variable: `y_factual`

**Output:**
```python
{
    "treatment": 0.45,
    "x1": 0.23,
    "x2": 0.18,
    ...
}
```

**Validation Checks:**
- All impact values should be non-negative (we use absolute values)
- Treatment variable should have above-median impact (sanity check)
- No NaN or infinite values

### Experiment 1B: Impact Value Ranges (Confidence Intervals)

**Logic:**
Point estimates are insufficient for risk-aware decision-making. We need uncertainty quantification:

1. **Bootstrap CI**: Resample data, recalculate impact, take percentiles
   - Distribution-free, makes no parametric assumptions
   - Captures sampling variability

2. **Analytical CI**: Use standard errors from regression
   - Faster but assumes normality
   - May underestimate uncertainty for small samples

**Conservative Estimation:**
Use the *lower bound* of the confidence interval as the impact value. This ensures:
- We don't overstate a variable's importance
- Selected variables have robust, reliable effects
- Reduces false positives in variable selection

**Output:**
```python
{
    "treatment": {"lower": 0.38, "point": 0.45, "upper": 0.52},
    "x1": {"lower": 0.15, "point": 0.23, "upper": 0.31},
    ...
}
```

---

## EXPERIMENT 2: Quadrant Division by Impact VALUE

### Objective
Partition variables into four groups (quadrants) where each quadrant contributes approximately 25% of the total causal impact.

### Why Value-Based Division?

**Traditional approach (count-based):**
- Divide N variables into 4 groups of N/4 each
- Problem: Ignores that impact is often highly skewed (Pareto principle)
- A few variables may dominate total impact

**Our approach (value-based):**
- Each quadrant contributes equal *cumulative impact*
- Q1 might have 2 variables, Q4 might have 15
- Reflects the true importance distribution

### Experiment 2A: Value-Based Quadrant Assignment

**Algorithm:**
```
1. Sort variables by impact (descending)
2. Calculate cumulative_impact as running sum
3. Calculate cumulative_percentage = cumulative_impact / total_impact
4. Assign quadrants:
   - Q1: cumulative_percentage ≤ 0.25
   - Q2: 0.25 < cumulative_percentage ≤ 0.50
   - Q3: 0.50 < cumulative_percentage ≤ 0.75
   - Q4: cumulative_percentage > 0.75
```

**Example:**
```
Variable | Impact | Cumulative | Cum% | Quadrant
---------|--------|------------|------|----------
var_A    | 30     | 30         | 30%  | Q1
var_B    | 20     | 50         | 50%  | Q2
var_C    | 15     | 65         | 65%  | Q2
var_D    | 10     | 75         | 75%  | Q3
var_E    | 8      | 83         | 83%  | Q4
var_F    | 7      | 90         | 90%  | Q4
...      | ...    | ...        | ...  | Q4
```

**Key Insight:** Q1 has 1 variable (30% impact), Q4 has many variables (25% combined impact)

**Output:**
```python
{
    "Q1": ["var_A"],
    "Q2": ["var_B", "var_C"],
    "Q3": ["var_D"],
    "Q4": ["var_E", "var_F", ...]
}
```

### Experiment 2B: Validate Quadrant Distribution

**Validation Checks:**
1. Sum of Q1 impacts ≈ 25% of total (±2% tolerance)
2. Sum of Q2 impacts ≈ 25% of total
3. Sum of Q3 impacts ≈ 25% of total
4. Sum of Q4 impacts ≈ 25% of total

**Visualization:**
- Bar chart: Quadrant vs. Total Impact (should be ~equal heights)
- Bar chart: Quadrant vs. Variable Count (will be unequal - expected)

---

## EXPERIMENT 3: Variable Classification by Causal Role

### Objective
Label each variable according to its structural role in the causal graph (PAG).

### Why Classify Variables?

Different causal roles have different implications:
- **Confounders** MUST be controlled to avoid bias
- **Mediators** should NOT be controlled (blocks causal pathway)
- **Edge variables** may or may not matter depending on context

Understanding roles helps interpret why certain variables are selected.

### Experiment 3A: Extract Variable Roles from PAG

**Definitions:**

1. **Independent Variables (Mediators)**
   - On a directed path from treatment → outcome
   - Mechanism through which treatment affects outcome
   - Graph pattern: `treatment → X → outcome`

2. **Confounders**
   - Common cause of both treatment AND outcome
   - Must be controlled to estimate causal effect
   - Graph pattern: `X → treatment` AND `X → outcome`

3. **Edge Variables**
   - Connected to the causal system but not on main pathway
   - Have edges in PAG (any type: o-o, o->, <->)
   - May have indirect effects

4. **Other**
   - Isolated variables or those not fitting above categories
   - Potentially irrelevant to causal question

**Algorithm:**
```python
def classify_variable(var, pag, treatment, outcome):
    # Check if on directed path treatment → outcome
    if is_on_directed_path(var, treatment, outcome, pag):
        return "independent"
    
    # Check if common cause of treatment and outcome
    if has_path_to(var, treatment, pag) and has_path_to(var, outcome, pag):
        return "confounder"
    
    # Check if has any edges in PAG
    if has_any_edges(var, pag):
        return "edge"
    
    return "other"
```

**Output:**
```python
{
    "treatment": "independent",
    "x1": "confounder",
    "x2": "edge",
    "x3": "independent",
    ...
}
```

### Experiment 3B: Cross-Reference Roles with Quadrants

**Purpose:** Understand the distribution of causal roles across impact levels

**Output Matrix:**
```
           | Independent | Confounder | Edge | Other |
-----------|-------------|------------|------|-------|
Q1 (High)  | var3, var7  | var1       |      |       |
Q2         | var9        | var4, var8 | var2 |       |
Q3         |             | var11      | var5 | var6  |
Q4 (Low)   | (excluded from selection)              |
```

**Insights:**
- High-impact confounders (Q1/Q2): Critical for valid inference
- High-impact mediators (Q1/Q2): Key mechanisms to understand
- Low-impact variables (Q4): Safe to exclude

---

## EXPERIMENT 4: 75% Selection Rule

### Objective
Select a parsimonious set of variables that collectively account for 75% of total causal impact, with balanced representation from Q1, Q2, and Q3.

### Why 75%?

**Trade-off considerations:**
- **Too low (e.g., 50%)**: May miss important effects, underestimate variance
- **Too high (e.g., 95%)**: Includes many low-impact variables, adds noise
- **75%**: Captures most variance while maintaining parsimony (Pareto-inspired)

### Why Equal Count from Each Quadrant?

**Problem with greedy selection:**
- Selecting top-k variables gives all Q1 variables
- Misses diversity and context from Q2/Q3
- May overfit to a few dominant variables

**Our approach:**
- Equal representation ensures balanced coverage
- Q2/Q3 variables may capture different aspects of the outcome
- More robust to model misspecification

### Experiment 4A: Equal Count Selection

**Algorithm:**
```
1. Exclude Q4 entirely (lowest 25% impact - diminishing returns)
2. Set target = 0.75 × total_impact
3. For n = 1, 2, 3, ...:
   a. Select top n variables from Q1
   b. Select top n variables from Q2
   c. Select top n variables from Q3
   d. Calculate total_selected_impact
   e. If total_selected_impact ≥ target:
      Return selected variables
4. Handle edge case: if a quadrant has fewer than n variables,
   take all available from that quadrant
```

**Example:**
```
Iteration | Q1 selected | Q2 selected | Q3 selected | Total Impact | Target
----------|-------------|-------------|-------------|--------------|--------
n=1       | 1 var (30%) | 1 var (20%) | 1 var (10%) | 60%          | 75%
n=2       | 2 var (35%) | 2 var (30%) | 2 var (15%) | 80%          | 75% ✓
```

**Output:**
```python
{
    "selected_variables": ["var_A", "var_B", "var_C", "var_D", "var_E", "var_F"],
    "total_impact_covered": 0.80,
    "count_per_quadrant": {"Q1": 2, "Q2": 2, "Q3": 2}
}
```

### Experiment 4B: Sensitivity Analysis on Threshold

**Purpose:** Validate that 75% is a reasonable choice (not arbitrary)

**Test thresholds:** 60%, 70%, 75%, 80%, 90%

**Metrics to track:**
- Number of variables selected
- Actual impact covered
- Variables per quadrant
- Computational cost

**Expected finding:** Diminishing returns after ~75% (many variables needed for last 25%)

---

## EXPERIMENT 5: Statistical Test Application

### Objective
Apply appropriate statistical tests to the selected variables to establish significance of their contribution to outcome variance.

### Why Test Selection Matters

Different data characteristics require different tests:
- **Normal data**: Parametric tests (more powerful if assumptions hold)
- **Non-normal data**: Non-parametric tests (robust but less powerful)
- **Small samples**: Tests designed for small n (t-test)
- **Large samples**: Tests leveraging asymptotic properties (z-test)

### Experiment 5A: Determine Distribution Types

**Classification Matrix:**

| Distribution | Sample Size | Classification |
|--------------|-------------|----------------|
| Normal       | n < 30      | Normal-Small   |
| Normal       | n ≥ 30      | Normal-Large   |
| Non-normal   | n < 30      | NonNormal-Small|
| Non-normal   | n ≥ 30      | NonNormal-Large|

**Normality Test:** Shapiro-Wilk (α = 0.05)
- H0: Data is normally distributed
- If p > 0.05: Assume normal
- If p ≤ 0.05: Assume non-normal

**Output:**
```python
{
    "var_A": {"classification": "Normal-Large", "shapiro_p": 0.23},
    "var_B": {"classification": "NonNormal-Small", "shapiro_p": 0.001},
    ...
}
```

### Experiment 5B: Apply Statistical Tests

**Test Selection Logic:**

| Classification    | Test Applied       | Effect Size Measure |
|-------------------|--------------------|--------------------|
| Normal-Small      | Independent t-test | Cohen's d          |
| Normal-Large      | Z-test             | Cohen's d          |
| NonNormal-Small   | Mann-Whitney U     | Rank-biserial r    |
| NonNormal-Large   | Wilcoxon Rank-Sum  | Rank-biserial r    |

**Group Creation Strategy:**
```
For each selected variable X:
1. Split data by median of X:
   - Group 1: observations where X ≤ median(X)
   - Group 2: observations where X > median(X)
2. Compare outcome (Y) distributions between groups
3. Record: test statistic, p-value, effect size
```

**Why Median Split?**
- Creates balanced groups (n/2 each)
- Non-parametric (doesn't assume X distribution)
- Tests whether high vs. low values of X lead to different outcomes

**Output:**
```python
{
    "var_A": {
        "test": "t-test",
        "statistic": 2.45,
        "p_value": 0.015,
        "effect_size": 0.42,
        "significant": True
    },
    ...
}
```

### Experiment 5C: Variance Contribution Analysis

**Alternative approach using model comparison:**

**Algorithm:**
```
1. Fit full model: Y ~ X1 + X2 + ... + Xn (all selected variables)
2. Record R²_full

3. For each variable Xi:
   a. Fit reduced model: Y ~ all variables EXCEPT Xi
   b. Record R²_reduced
   c. Calculate ΔR² = R²_full - R²_reduced
   d. Test significance using F-test

4. Rank variables by ΔR²
```

**Interpretation:**
- Large ΔR²: Variable is irreplaceable, removing it significantly degrades model
- Small ΔR²: Variable's contribution is captured by other variables (redundant)

**Why Both 5B and 5C?**
- 5B: Univariate effect (variable in isolation)
- 5C: Multivariate contribution (variable in context)
- Variables important in both are most reliable

---

## EXPERIMENT 6: Range Calculation for Outcome

### Objective
Aggregate the statistical results to produce a final range estimate for the outcome variable.

### Experiment 6A: Variance Aggregation (Sum of Squares)

**Logic:**
If selected variables explain independent portions of variance:
- Total variance explained ≈ Σ (individual variance contributions)
- Prediction interval width depends on √(total variance)

**Algorithm:**
```
1. For each significant variable (p < 0.05):
   - Calculate σ²_i (variance contribution from effect size)
   
2. Aggregate: σ²_total = Σ σ²_i

3. Calculate prediction interval:
   - Mean outcome: μ = mean(Y)
   - 95% interval: [μ - 1.96√σ²_total, μ + 1.96√σ²_total]
```

**Output:**
```
Outcome Range (95% CI): [3.42, 7.89]
Mean: 5.66
Total variance explained: 68%
```

### Experiment 6B: Bootstrap Range Estimation

**Logic:**
Account for uncertainty in the entire pipeline, not just final estimates.

**Algorithm:**
```
For b = 1 to 1000:
    1. Resample data with replacement
    2. Re-run Experiment 1 (impact values)
    3. Re-run Experiment 2 (quadrants)
    4. Re-run Experiment 4 (selection)
    5. Calculate predicted outcome range
    6. Store: [lower_b, upper_b]

Final range: [2.5th percentile of lowers, 97.5th percentile of uppers]
```

**Advantages:**
- Captures uncertainty from all pipeline stages
- No distributional assumptions
- More conservative (usually wider) than 6A

### Experiment 6C: Compare Range Estimation Methods

**Methods to compare:**
1. Sum of squares aggregation (6A)
2. Bootstrap method (6B)
3. Direct regression confidence interval
4. PAG-based identifiability bounds (if available)

**Evaluation criteria:**
- Width of interval (narrower is better, if calibrated)
- Coverage probability (should be ≈ 95% for 95% CI)
- Computational cost

---

## EXPERIMENT 7: Validation Against Ground Truth

### Objective
Assess whether our estimated ranges are calibrated and accurate.

### Experiment 7A: Coverage Analysis

**IHDP Advantage:** Semi-synthetic dataset with known treatment effects

**Metrics:**
1. **Coverage probability**: % of true values falling within estimated range
   - Target: 95% for a 95% CI
   - <95%: Interval too narrow (overconfident)
   - >95%: Interval too wide (underconfident)

2. **Mean Absolute Error (MAE)**: Average distance from point estimate to truth

3. **Interval Score**: Penalizes both width and miscoverage

**Algorithm:**
```
1. Run full pipeline → get [lower, upper] range
2. Compare against true outcome values
3. Calculate:
   - coverage = mean(lower ≤ Y_true ≤ upper)
   - MAE = mean(|Y_predicted - Y_true|)
```

### Experiment 7B: Subgroup Analysis

**Purpose:** Check if method works across different subpopulations

**Subgroups to test:**
- Low vs. high birth weight
- Treatment vs. control group
- Different demographic segments

**Expected insight:** Ranges may be wider for heterogeneous subgroups

---

## EXPERIMENT 8: Sensitivity Analysis

### Objective
Assess robustness of results to methodological choices.

### Experiment 8A: Quadrant Threshold Sensitivity

**Test configurations:**
- [20%, 40%, 60%, 80%] - Different boundary locations
- [25%, 50%, 75%, 100%] - Default (equal quarters)
- [10%, 30%, 60%, 100%] - Front-loaded

**Question:** How much do final ranges change with different quadrant definitions?

### Experiment 8B: Selection Percentage Sensitivity

**Test values:** 50%, 60%, 70%, 75%, 80%, 90%

**Expected relationship:**
- More variables selected → potentially wider range (more variance captured)
- But also more noise → range may not improve after threshold

**Goal:** Find optimal selection percentage (or confirm 75% is reasonable)

---

## EXPERIMENT 9: Visualization Dashboard

### Objective
Create interpretable visualizations for stakeholder communication.

### Experiment 9A: Impact Distribution Plot
- X-axis: Variables (sorted by impact)
- Y-axis: Impact value
- Color coding: Quadrant assignment
- Markers: Variable type (independent/confounder/edge)
- Vertical lines: Quadrant boundaries

### Experiment 9B: Selection Sankey Diagram
- Flow visualization: All vars → Quadrants → Selected → Outcome
- Width proportional to impact
- Shows how impact "flows" through selection process

### Experiment 9C: Range Comparison Chart
- Horizontal bars for each estimation method
- Ground truth range (if available)
- Multiple confidence levels (50%, 75%, 95%)

---

## EXPERIMENT 10: End-to-End Pipeline

### Objective
Integrate all experiments into a single automated workflow.

### Pipeline Steps
```
1. load_data()           → IHDP data + PAG
2. calculate_impacts()   → Experiment 1
3. assign_quadrants()    → Experiment 2
4. classify_roles()      → Experiment 3
5. select_variables()    → Experiment 4
6. run_tests()           → Experiment 5
7. calculate_range()     → Experiment 6
8. validate()            → Experiment 7
9. visualize()           → Experiment 9
```

### Final Output Format
```
═══════════════════════════════════════════════════════════
VARIANCE-BASED RANGE ESTIMATION RESULTS
═══════════════════════════════════════════════════════════

Outcome Variable: y_factual
Selected Variables: 6 (from 25 total)
Impact Coverage: 78.3%

OUTCOME RANGE (95% CI): [3.42, 7.89]
Point Estimate: 5.66

Selected Variables:
  Q1: treatment (impact=0.45), x3 (impact=0.32)
  Q2: x7 (impact=0.21), x1 (impact=0.18)
  Q3: x12 (impact=0.11), x5 (impact=0.09)

Validation Metrics:
  Coverage: 94.2%
  MAE: 0.34

═══════════════════════════════════════════════════════════
```

---

## Implementation Priority

### Phase 1: Core Functionality (Must-Have)
| Priority | Experiment | Description |
|----------|------------|-------------|
| 1 | 1A | Impact value calculation |
| 2 | 2A | Value-based quadrant division |
| 3 | 4A | 75% selection rule |
| 4 | 5B | Statistical tests |
| 5 | 6A | Range calculation |

### Phase 2: Validation (Important)
| Priority | Experiment | Description |
|----------|------------|-------------|
| 6 | 7A | Ground truth comparison |
| 7 | 3A | Variable classification |
| 8 | 2B | Quadrant validation |

### Phase 3: Robustness (Nice-to-Have)
| Priority | Experiment | Description |
|----------|------------|-------------|
| 9 | 8A | Quadrant threshold sensitivity |
| 10 | 8B | Selection rule sensitivity |
| 11 | 6B | Bootstrap ranges |

### Phase 4: Presentation (Final Polish)
| Priority | Experiment | Description |
|----------|------------|-------------|
| 12 | 9A | Impact distribution plot |
| 13 | 9B | Selection Sankey diagram |
| 14 | 9C | Range comparison chart |
| 15 | 10 | Integrated pipeline |

---

## Success Criteria

After completing all experiments, the system should answer:

| Question | Experiment | Expected Answer |
|----------|------------|-----------------|
| Does the method work? | 7A | Coverage ≈ 95% |
| Why 75%? | 8B | Diminishing returns analysis |
| Which variables matter? | 2A, 3A | Ranked list with roles |
| How wide is uncertainty? | 6A/6B | Quantified range |
| Is it better than alternatives? | 6C | Method comparison |

---

## Appendix: Mathematical Foundations

### Causal Effect Estimation
The Average Treatment Effect (ATE) under backdoor adjustment:
```
ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
    = Σ_z (E[Y | X=1, Z=z] - E[Y | X=0, Z=z]) P(Z=z)
```

### Variance Decomposition
Total outcome variance can be decomposed:
```
Var(Y) = Σ_i Var(Y | X_i) + Residual
```

Our method captures Σ_i Var(Y | X_i) for selected variables accounting for 75% of impact.

### Confidence Interval Construction
For 95% CI with estimated variance σ²:
```
CI = [μ - 1.96σ, μ + 1.96σ]
```

For bootstrap CI:
```
CI = [percentile_2.5(bootstrap_estimates), percentile_97.5(bootstrap_estimates)]
```
