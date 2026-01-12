"""
Experiment 3A: Extract Variable Roles from PAG

This script classifies variables by their causal role in the PAG:
- Independent: On directed path from treatment → outcome
- Confounder: Has paths to BOTH treatment AND outcome
- Edge: Has edges in PAG but not on treatment → outcome path
- Other: Doesn't fit above categories
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiment_1'))

from experiment_1.utils.data_loader import load_fci_pag, get_variable_names, get_outcome_variable, get_treatment_variable
from experiment_1.utils.pag_analysis import (
    get_causal_paths, 
    find_parents, 
    is_ancestor,
    get_directly_connected_vars
)


def find_all_paths_between(pag: np.ndarray, source: int, target: int, max_length: int = 10) -> List[List[int]]:
    """
    Find all paths (directed or undirected) between source and target.
    
    Parameters
    ----------
    pag : np.ndarray
        PAG adjacency matrix
    source : int
        Source node
    target : int
        Target node
    max_length : int
        Maximum path length
        
    Returns
    -------
    list
        List of paths
    """
    paths = []
    
    def dfs(current: int, path: List[int], visited: Set[int], depth: int):
        if depth > max_length:
            return
            
        if current == target and len(path) > 1:
            paths.append(path.copy())
            return
        
        visited.add(current)
        
        # Find all connected nodes (any edge type)
        for neighbor in range(pag.shape[0]):
            if neighbor in visited or neighbor == current:
                continue
            
            # Check if there's any edge
            if pag[current, neighbor] != 0 or pag[neighbor, current] != 0:
                path.append(neighbor)
                dfs(neighbor, path, visited, depth + 1)
                path.pop()
        
        visited.remove(current)
    
    dfs(source, [source], set(), 0)
    return paths


def classify_variable_role(pag: np.ndarray, var_idx: int, treatment_idx: int, 
                          outcome_idx: int, var_names: List[str]) -> str:
    """
    Classify a variable's causal role in the PAG.
    
    Parameters
    ----------
    pag : np.ndarray
        PAG adjacency matrix
    var_idx : int
        Variable index to classify
    treatment_idx : int
        Treatment variable index
    outcome_idx : int
        Outcome variable index
    var_names : list
        Variable names
        
    Returns
    -------
    str
        Role: 'independent', 'confounder', 'edge', or 'other'
    """
    # Skip if this is treatment or outcome
    if var_idx == treatment_idx or var_idx == outcome_idx:
        return 'outcome' if var_idx == outcome_idx else 'treatment'
    
    # Get directed paths from treatment to outcome
    treatment_to_outcome_paths = get_causal_paths(pag, treatment_idx, outcome_idx)
    
    # Check if variable is on any treatment → outcome path (Independent/Mediator)
    on_treatment_outcome_path = False
    for path in treatment_to_outcome_paths:
        if var_idx in path:
            on_treatment_outcome_path = True
            break
    
    if on_treatment_outcome_path:
        return 'independent'
    
    # Check for paths to treatment and outcome (Confounder)
    paths_to_treatment = find_all_paths_between(pag, var_idx, treatment_idx, max_length=5)
    paths_to_outcome = find_all_paths_between(pag, var_idx, outcome_idx, max_length=5)
    
    has_path_to_treatment = len(paths_to_treatment) > 0
    has_path_to_outcome = len(paths_to_outcome) > 0
    
    if has_path_to_treatment and has_path_to_outcome:
        return 'confounder'
    
    # Check if variable has any edges in PAG (Edge variable)
    connected_vars = get_directly_connected_vars(pag, var_idx)
    
    if len(connected_vars) > 0:
        return 'edge'
    
    # Otherwise, it's isolated or other
    return 'other'


def classify_all_variables(pag: np.ndarray, var_names: List[str], 
                          treatment_var: str, outcome_var: str) -> Dict[str, str]:
    """
    Classify all variables by their causal role.
    
    Parameters
    ----------
    pag : np.ndarray
        PAG adjacency matrix
    var_names : list
        Variable names
    treatment_var : str
        Treatment variable name
    outcome_var : str
        Outcome variable name
        
    Returns
    -------
    dict
        Dictionary of {variable: role}
    """
    treatment_idx = var_names.index(treatment_var)
    outcome_idx = var_names.index(outcome_var)
    
    role_dict = {}
    role_counts = {'independent': 0, 'confounder': 0, 'edge': 0, 'other': 0, 
                   'treatment': 0, 'outcome': 0}
    
    print("Classifying variables by causal role...")
    print()
    
    for i, var in enumerate(var_names):
        role = classify_variable_role(pag, i, treatment_idx, outcome_idx, var_names)
        role_dict[var] = role
        role_counts[role] += 1
        
        print(f"  {var}: {role}")
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    for role, count in sorted(role_counts.items()):
        print(f"  {role.capitalize()}: {count} variables")
    print("=" * 60)
    
    return role_dict


def save_classification_results(role_dict: Dict[str, str], output_dir: str = 'results'):
    """
    Save classification results to files.
    
    Parameters
    ----------
    role_dict : dict
        Variable role dictionary
    output_dir : str
        Output directory
    """
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save as JSON
    json_path = output_path / 'variable_roles.json'
    with open(json_path, 'w') as f:
        json.dump(role_dict, f, indent=2)
    print(f"\n✓ Saved roles (JSON): {json_path}")
    
    # Save as CSV
    csv_path = output_path / 'variable_roles.csv'
    df = pd.DataFrame(list(role_dict.items()), columns=['variable', 'role'])
    df = df.sort_values(['role', 'variable'])
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved roles (CSV): {csv_path}")
    
    # Save grouped by role
    grouped_path = output_path / 'variables_by_role.json'
    grouped = {}
    for var, role in role_dict.items():
        if role not in grouped:
            grouped[role] = []
        grouped[role].append(var)
    
    with open(grouped_path, 'w') as f:
        json.dump(grouped, f, indent=2)
    print(f"✓ Saved grouped roles: {grouped_path}")


def run_experiment_3a():
    """
    Run Experiment 3A: Extract Variable Roles from PAG
    """
    print("=" * 80)
    print("EXPERIMENT 3A: EXTRACT VARIABLE ROLES FROM PAG")
    print("=" * 80)
    print()
    
    # Load PAG and variable names
    print("Loading PAG and variables...")
    pag = load_fci_pag()
    var_names = get_variable_names()
    treatment_var = get_treatment_variable()
    outcome_var = get_outcome_variable()
    
    print(f"Treatment: {treatment_var}")
    print(f"Outcome: {outcome_var}")
    print(f"Total variables: {len(var_names)}")
    print()
    
    # Classify all variables
    role_dict = classify_all_variables(pag, var_names, treatment_var, outcome_var)
    
    # Save results
    save_classification_results(role_dict)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 3A COMPLETED")
    print("=" * 80)
    
    return role_dict


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 3A")
    print("=" * 80)
    print()
    
    # Run experiment
    results = run_experiment_3a()
    
    print("\n✓ Variable classification completed successfully!")
