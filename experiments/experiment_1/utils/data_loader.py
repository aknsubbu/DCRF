"""
Data loading utilities for Experiment 1
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    """Get the project root directory (DCRF folder)"""
    current_file = Path(__file__)
    # Navigate up: data_loader.py -> utils -> experiment_1 -> experiments -> DCRF
    return current_file.parent.parent.parent.parent


def load_ihdp_data():
    """
    Load the IHDP dataset.
    
    Returns
    -------
    pd.DataFrame
        IHDP dataset with all variables
    """
    project_root = get_project_root()
    data_path = project_root / 'data' / 'ihdp.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"IHDP data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Convert treatment to numeric if it's boolean
    if df['treatment'].dtype == bool:
        df['treatment'] = df['treatment'].astype(int)
    
    print(f"✓ Loaded IHDP data: {df.shape[0]} samples × {df.shape[1]} variables")
    
    return df


def load_fci_pag():
    """
    Load the FCI PAG (Partial Ancestral Graph) adjacency matrix.
    
    Returns
    -------
    np.ndarray
        Adjacency matrix of the PAG
        Encoding: 0 = no edge, 1 = arrowhead (->), 2 = tail (--), -1 = circle (-o)
    """
    project_root = get_project_root()
    pag_path = project_root / 'fci_adjacency_matrix.csv'
    
    if not pag_path.exists():
        raise FileNotFoundError(f"FCI PAG not found at {pag_path}")
    
    # Load as numpy array
    pag = pd.read_csv(pag_path, header=None).values
    
    print(f"✓ Loaded FCI PAG: {pag.shape[0]}×{pag.shape[1]} adjacency matrix")
    
    return pag


def get_variable_names():
    """
    Get the variable names from the IHDP dataset.
    
    Returns
    -------
    list
        List of variable names in the same order as PAG matrix
    """
    df = load_ihdp_data()
    variable_names = df.columns.tolist()
    
    return variable_names


def get_outcome_variable():
    """
    Get the name of the outcome variable.
    
    Returns
    -------
    str
        Name of the outcome variable ('y_factual')
    """
    return 'y_factual'


def get_treatment_variable():
    """
    Get the name of the treatment variable.
    
    Returns
    -------
    str
        Name of the treatment variable ('treatment')
    """
    return 'treatment'


def load_data_and_pag():
    """
    Convenience function to load both IHDP data and FCI PAG.
    
    Returns
    -------
    tuple
        (data DataFrame, PAG adjacency matrix, variable names)
    """
    data = load_ihdp_data()
    pag = load_fci_pag()
    var_names = data.columns.tolist()
    
    # Validate dimensions match
    if len(var_names) != pag.shape[0]:
        raise ValueError(
            f"Mismatch: data has {len(var_names)} variables "
            f"but PAG has {pag.shape[0]} nodes"
        )
    
    return data, pag, var_names


if __name__ == "__main__":
    # Test loading
    print("Testing data loader...")
    print()
    
    data, pag, var_names = load_data_and_pag()
    
    print(f"\nVariable names ({len(var_names)}):")
    print(var_names)
    
    print(f"\nOutcome variable: {get_outcome_variable()}")
    print(f"Treatment variable: {get_treatment_variable()}")
    
    print(f"\nPAG shape: {pag.shape}")
    print(f"Number of edges in PAG: {np.sum(pag != 0)}")
