"""
PAG (Partial Ancestral Graph) analysis utilities for identifying causal relationships
and adjustment sets.
"""

import numpy as np
from typing import List, Set, Tuple, Optional


def find_parents(pag: np.ndarray, node: int) -> List[int]:
    """
    Find parent nodes (nodes with directed edges into the given node).
    
    A parent exists when pag[parent, node] = 1 (arrowhead) and pag[node, parent] = 2 (tail)
    This represents: parent -> node
    
    Parameters
    ----------
    pag : np.ndarray
        PAG adjacency matrix
    node : int
        Node index
        
    Returns
    -------
    List[int]
        List of parent node indices
    """
    n_vars = pag.shape[0]
    parents = []
    
    for i in range(n_vars):
        if i == node:
            continue
        # Check if i -> node (i has tail, node has arrowhead)
        if pag[i, node] == 1 and pag[node, i] == 2:
            parents.append(i)
    
    return parents


def find_children(pag: np.ndarray, node: int) -> List[int]:
    """
    Find children nodes (nodes with directed edges from the given node).
    
    A child exists when pag[node, child] = 1 (arrowhead) and pag[child, node] = 2 (tail)
    This represents: node -> child
    
    Parameters
    ----------
    pag : np.ndarray
        PAG adjacency matrix
    node : int
        Node index
        
    Returns
    -------
    List[int]
        List of child node indices
    """
    n_vars = pag.shape[0]
    children = []
    
    for j in range(n_vars):
        if j == node:
            continue
        # Check if node -> j (node has tail, j has arrowhead)
        if pag[node, j] == 1 and pag[j, node] == 2:
            children.append(j)
    
    return children


def is_ancestor(pag: np.ndarray, ancestor: int, descendant: int, 
                visited: Optional[Set[int]] = None) -> bool:
    """
    Check if one node is an ancestor of another using recursive search.
    
    Parameters
    ----------
    pag : np.ndarray
        PAG adjacency matrix
    ancestor : int
        Potential ancestor node
    descendant : int
        Potential descendant node
    visited : Set[int], optional
        Nodes already visited (for recursion)
        
    Returns
    -------
    bool
        True if ancestor is an ancestor of descendant
    """
    if visited is None:
        visited = set()
    
    if ancestor == descendant:
        return True
    
    if ancestor in visited:
        return False
    
    visited.add(ancestor)
    
    # Get children of ancestor
    children = find_children(pag, ancestor)
    
    for child in children:
        if child == descendant or is_ancestor(pag, child, descendant, visited):
            return True
    
    return False


def get_causal_paths(pag: np.ndarray, source: int, target: int) -> List[List[int]]:
    """
    Find all potentially causal paths from source to target.
    
    A causal path is a directed path following arrows in the PAG.
    
    Parameters
    ----------
    pag : np.ndarray
        PAG adjacency matrix
    source : int
        Source node
    target : int
        Target node
        
    Returns
    -------
    List[List[int]]
        List of paths, where each path is a list of node indices
    """
    paths = []
    
    def dfs(current: int, path: List[int], visited: Set[int]):
        if current == target:
            paths.append(path.copy())
            return
        
        visited.add(current)
        children = find_children(pag, current)
        
        for child in children:
            if child not in visited:
                path.append(child)
                dfs(child, path, visited)
                path.pop()
        
        visited.remove(current)
    
    dfs(source, [source], set())
    return paths


def find_adjustment_set(pag: np.ndarray, treatment: int, outcome: int, 
                       var_names: Optional[List[str]] = None) -> List[int]:
    """
    Find a valid adjustment set for estimating the causal effect of treatment on outcome.
    
    This uses a simplified backdoor criterion for PAGs:
    1. Include all parents of treatment (confounders)
    2. Exclude descendants of treatment
    3. Exclude colliders (nodes where two arrows meet)
    
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
    n_vars = pag.shape[0]
    adjustment_set = set()
    
    # Get parents of treatment (potential confounders)
    treatment_parents = find_parents(pag, treatment)
    adjustment_set.update(treatment_parents)
    
    # Get parents of outcome that are not treatment or descendants of treatment
    outcome_parents = find_parents(pag, outcome)
    for parent in outcome_parents:
        if parent != treatment and not is_ancestor(pag, treatment, parent):
            adjustment_set.add(parent)
    
    # Ensure we don't include outcome or treatment in adjustment set
    adjustment_set.discard(treatment)
    adjustment_set.discard(outcome)
    
    # Remove any descendants of treatment (mediators)
    descendants_to_remove = set()
    for node in adjustment_set:
        if is_ancestor(pag, treatment, node):
            descendants_to_remove.add(node)
    
    adjustment_set -= descendants_to_remove
    
    adjustment_list = sorted(list(adjustment_set))
    
    if var_names:
        print(f"Adjustment set for {var_names[treatment]} -> {var_names[outcome]}:")
        if adjustment_list:
            print(f"  Adjust for: {[var_names[i] for i in adjustment_list]}")
        else:
            print(f"  No adjustment needed (or no valid adjustment found)")
    
    return adjustment_list


def get_directly_connected_vars(pag: np.ndarray, node: int) -> List[int]:
    """
    Get all variables directly connected to the given node (any type of edge).
    
    Parameters
    ----------
    pag : np.ndarray
        PAG adjacency matrix
    node : int
        Node index
        
    Returns
    -------
    List[int]
        List of directly connected node indices
    """
    n_vars = pag.shape[0]
    connected = []
    
    for i in range(n_vars):
        if i == node:
            continue
        # Any non-zero edge
        if pag[node, i] != 0 or pag[i, node] != 0:
            connected.append(i)
    
    return connected


def summarize_pag_structure(pag: np.ndarray, var_names: Optional[List[str]] = None):
    """
    Print a summary of the PAG structure.
    
    Parameters
    ----------
    pag : np.ndarray
        PAG adjacency matrix
    var_names : List[str], optional
        Variable names
    """
    n_vars = pag.shape[0]
    
    if var_names is None:
        var_names = [f"X{i}" for i in range(n_vars)]
    
    print("=" * 60)
    print("PAG Structure Summary")
    print("=" * 60)
    print(f"Number of variables: {n_vars}")
    
    # Count edge types
    n_directed = 0
    n_undirected = 0
    n_partial = 0
    
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if pag[i, j] != 0 or pag[j, i] != 0:
                # Directed edge: i -> j
                if pag[i, j] == 1 and pag[j, i] == 2:
                    n_directed += 1
                # Directed edge: j -> i
                elif pag[j, i] == 1 and pag[i, j] == 2:
                    n_directed += 1
                # Undirected edge: i -- j
                elif pag[i, j] == 2 and pag[j, i] == 2:
                    n_undirected += 1
                # Partially directed (has circle)
                else:
                    n_partial += 1
    
    print(f"Directed edges (->): {n_directed}")
    print(f"Undirected edges (--): {n_undirected}")
    print(f"Partially directed (-o): {n_partial}")
    print(f"Total edges: {n_directed + n_undirected + n_partial}")
    print("=" * 60)
    
    # Show parents for each variable
    print("\nParent relationships:")
    for i in range(n_vars):
        parents = find_parents(pag, i)
        if parents:
            parent_names = [var_names[p] for p in parents]
            print(f"  {var_names[i]}: parents = {parent_names}")


if __name__ == "__main__":
    # Test with a simple synthetic PAG
    print("Testing PAG analysis utilities...")
    print()
    
    # Create a simple test PAG: X0 -> X1 -> X2, X0 -> X2
    test_pag = np.zeros((3, 3), dtype=int)
    # X0 -> X1
    test_pag[0, 1] = 1  # arrowhead at X1
    test_pag[1, 0] = 2  # tail at X0
    # X1 -> X2
    test_pag[1, 2] = 1  # arrowhead at X2
    test_pag[2, 1] = 2  # tail at X1
    # X0 -> X2
    test_pag[0, 2] = 1  # arrowhead at X2
    test_pag[2, 0] = 2  # tail at X0
    
    var_names = ['X0', 'X1', 'X2']
    
    summarize_pag_structure(test_pag, var_names)
    
    print("\n" + "=" * 60)
    print("Testing specific functions:")
    print("=" * 60)
    
    print(f"\nParents of X2: {[var_names[i] for i in find_parents(test_pag, 2)]}")
    print(f"Children of X0: {[var_names[i] for i in find_children(test_pag, 0)]}")
    print(f"Is X0 ancestor of X2? {is_ancestor(test_pag, 0, 2)}")
    print(f"Causal paths X0 -> X2: {get_causal_paths(test_pag, 0, 2)}")
    
    print()
    adjustment_set = find_adjustment_set(test_pag, 0, 2, var_names)
