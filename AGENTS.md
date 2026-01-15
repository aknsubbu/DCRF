# AGENTS.md - Coding Agent Instructions for DCRF

## Project Overview

**Dynamic Causal Reasoning Framework (DCRF)** - A Python toolkit for causal discovery, analysis, and reasoning. The framework implements FCI (Fast Causal Inference) algorithm for causal structure learning, multiple causal effect estimation methods, and statistical validation pipelines.

**Tech Stack:** 
- **Backend/Analysis:** Python 3.8+, pandas, numpy, scipy, scikit-learn, causal-learn, networkx, statsmodels, matplotlib, seaborn
- **Frontend:** Next.js 15, React 18, TypeScript, React Flow, HeroUI, Tailwind CSS 4

## Build/Run Commands

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn
pip install causal-learn networkx statsmodels
```

### Running Experiments
```bash
# Run individual experiment scripts
python experiments/experiment_1/run_experiment_1a.py
python experiments/experiment_1/run_experiment_1b.py
python experiments/experiment_2/run_experiment_2a.py
# ... and so on for experiment_3, experiment_4, experiment_5

# Run Jupyter notebooks
jupyter notebook fci.ipynb
jupyter notebook causal.ipynb
```

### Running Frontend
```bash
cd frontend
npm install
npm run dev     # Development server with Turbopack
npm run build   # Production build
npm run lint    # ESLint with auto-fix
```

### Testing
**No formal test framework is configured.** Testing is done via:
1. Inline `if __name__ == "__main__":` blocks with test code at module bottom
2. `validate_results()` functions in experiment scripts
3. Manual verification through notebooks

```bash
# Run a module's built-in tests
python experiments/experiment_1/utils/effect_estimators.py
python experiments/experiment_1/utils/pag_analysis.py
```

## Code Style Guidelines

### File Structure
- Experiments are organized in `experiments/experiment_N/` directories
- Each experiment has run scripts (`run_experiment_Na.py`, `run_experiment_Nb.py`)
- Shared utilities go in `utils/` subdirectories with `__init__.py`
- Results are saved to `results/` subdirectories (CSV and JSON formats)

### Imports
```python
# Standard library first
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

# Third-party second
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Local imports last (with path manipulation when needed)
sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import load_data_and_pag
from utils.effect_estimators import LinearRegressionEstimator
```

### Naming Conventions
- **Functions/variables:** `snake_case` (e.g., `find_adjustment_set`, `treatment_var`)
- **Classes:** `PascalCase` (e.g., `LinearRegressionEstimator`, `BaseEstimator`)
- **Constants:** `UPPER_SNAKE_CASE` (rarely used)
- **Files:** `snake_case.py` for modules, `run_experiment_Na.py` for scripts
- **Directories:** `lowercase` (e.g., `experiment_1`, `utils`, `results`)

### Docstrings - NumPy Style
```python
def find_adjustment_set(pag: np.ndarray, treatment: int, outcome: int,
                        var_names: Optional[List[str]] = None) -> List[int]:
    """
    Find a valid adjustment set for estimating causal effect.
    
    Parameters
    ----------
    pag : np.ndarray
        PAG adjacency matrix
    treatment : int
        Treatment variable index
    outcome : int
        Outcome variable index
    var_names : List[str], optional
        Variable names for better output
        
    Returns
    -------
    List[int]
        List of variable indices to adjust for
    """
```

### Type Hints
Use type hints from `typing` module for function signatures:
```python
from typing import List, Dict, Optional, Tuple, Set

def process_data(data: pd.DataFrame, variables: List[str]) -> Dict[str, float]:
    ...
```

### Class Design
- Use abstract base classes with `@abstractmethod` for estimator interfaces
- Call `super().__init__()` in subclass constructors
- Track state with instance variables (e.g., `self.is_fitted`)

```python
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    def __init__(self):
        self.is_fitted = False
        self.impact_value = None
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, treatment: str, outcome: str,
            adjustment_set: List[str]) -> 'BaseEstimator':
        pass
```

### Error Handling
```python
# Use try/except with warnings for non-critical errors
try:
    result = process_variable(var)
except Exception as e:
    if verbose:
        print(f"  Warning: Error processing {var}: {str(e)}")
    continue

# Use RuntimeError for state errors
if not self.is_fitted:
    raise RuntimeError("Model not fitted. Call fit() first.")

# Use ValueError for invalid arguments
if self.model_type not in ['rf', 'gb']:
    raise ValueError(f"Unknown model_type: {self.model_type}")
```

### Warnings
```python
import warnings
warnings.filterwarnings('ignore')  # At top of scripts

# Or specific warnings
warnings.warn(
    "CausalForestEstimator is using Random Forest variable importance "
    "as a proxy. Install econml for true causal forest implementation."
)
```

### Output and Progress
Use visual indicators for user feedback:
```python
print("=" * 80)
print("EXPERIMENT 1A: DIRECT CAUSAL EFFECT ESTIMATION")
print("=" * 80)

print(f"[{i+1}/{total}] Processing variable: {var}")
print(f"  - Linear Regression... Impact: {impact:.4f}")
print(f"✓ Results saved to: {csv_path}")
print(f"⚠ Warning: Found {count} NaN values")
```

### Reproducibility
Always set random seeds for reproducible results:
```python
np.random.seed(42)
random_state = 42

# In sklearn models
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
```

### Data Handling
```python
# Load data
data = pd.read_csv(path)

# Save results - both CSV and JSON
results_df.to_csv(csv_path, index=False)

# Convert numpy types for JSON serialization
for key, val in result.items():
    if isinstance(val, (np.bool_, np.integer, np.floating)):
        result[key] = val.item()

with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
```

### Directory Creation
```python
from pathlib import Path

output_path = Path(__file__).parent / 'results'
output_path.mkdir(exist_ok=True, parents=True)
```

## Key Domain Concepts

- **PAG (Partial Ancestral Graph):** Causal graph output from FCI algorithm
- **Adjustment set:** Variables to control for when estimating causal effects
- **Impact value:** Quantified causal effect of a variable on the outcome
- **Quadrants:** Variables grouped by cumulative impact contribution (Q1=highest)
- **IHDP dataset:** Benchmark dataset in `data/ihdp.csv` with ~747 samples, 25 variables

## Important Files

| Path | Description |
|------|-------------|
| `data/ihdp.csv` | Main dataset (Infant Health and Development Program) |
| `fci_adjacency_matrix.csv` | FCI-generated PAG adjacency matrix |
| `experiments/experiment_N/` | Experiment implementations (N=1-5) |
| `experiments/experiment_1/utils/` | Shared utilities for data loading, PAG analysis, effect estimation |
| `frontend/` | Next.js causal graph visualization app |
| `docs/` | Documentation including literature survey and experiment walkthroughs |

## Script Entry Points
All experiment scripts follow this pattern:
```python
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT XY")
    print("=" * 80)
    
    results_df = run_experiment_xy(verbose=True)
    validation_passed = validate_results(results_df, verbose=True)
    save_results(results_df)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT XY COMPLETED")
    print("=" * 80)
```

## Frontend Development

The frontend is a Next.js 15 application with React Flow for graph visualization.

### Frontend File Structure
```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx          # Root layout with providers
│   ├── page.tsx            # Main visualization page
│   └── providers.tsx       # HeroUI + Theme providers
├── components/
│   └── graph/              # Graph visualization components
│       ├── GraphCanvas.tsx # React Flow canvas wrapper
│       ├── TerminalNode.tsx# Custom styled node component
│       ├── UploadSection.tsx# CSV upload with dropzone
│       ├── PathFinderPanel.tsx# Path search controls
│       └── ...
├── lib/                    # Utility functions
│   ├── graphUtils.ts       # Graph conversion & layout
│   └── pathFinding.ts      # BFS path algorithms
├── types/                  # TypeScript type definitions
└── styles/
    └── globals.css         # Global styles & terminal theme
```

### Frontend Conventions
- **Components:** PascalCase in `components/` directory
- **Utilities:** camelCase in `lib/` directory  
- **Types:** PascalCase interfaces in `types/` directory
- **Styling:** Tailwind CSS 4 utility classes, terminal/CRT theme
- **State:** React hooks, no external state management
