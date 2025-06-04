"""
High-level Python interface for Leiden and Louvain community detection algorithms.

This module provides pandas DataFrame and Apache Arrow compatible interfaces
for the high-performance C++ implementations.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa


def _validate_graph_dataframe(df: pd.DataFrame) -> None:
    """Validate input DataFrame for graph data."""
    required_cols = ["source", "target"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")

    if df.empty:
        raise ValueError("DataFrame cannot be empty")

    # Check for invalid values
    if df[["source", "target"]].isnull().any().any():
        raise ValueError("Source and target columns cannot contain NaN values")

    if (df[["source", "target"]] < 0).any().any():
        raise ValueError("Source and target vertex IDs must be non-negative")


def _prepare_graph_arrays(df: pd.DataFrame) -> tuple:
    """Convert DataFrame to numpy arrays for C++ bindings."""
    # Ensure vertex IDs are contiguous starting from 0
    vertices = pd.concat([df["source"], df["target"]]).unique()
    vertices.sort()

    # Create mapping for non-contiguous vertex IDs
    vertex_map = {v: i for i, v in enumerate(vertices)}
    original_vertices = vertices

    # Map vertex IDs to contiguous range
    sources = df["source"].map(vertex_map).astype(np.uint32)
    targets = df["target"].map(vertex_map).astype(np.uint32)

    # Handle weights
    if "weight" in df.columns:
        weights = df["weight"].fillna(1.0).astype(np.float64)
    else:
        weights = np.ones(len(df), dtype=np.float64)

    # Extract values, handling both pandas Series and numpy arrays
    sources_vals = sources.values if hasattr(sources, "values") else sources
    targets_vals = targets.values if hasattr(targets, "values") else targets
    weights_vals = weights.values if hasattr(weights, "values") else weights

    return sources_vals, targets_vals, weights_vals, original_vertices, vertex_map


def _convert_result_to_dataframe(
    result: Dict[str, Any], original_vertices: np.ndarray
) -> Dict[str, Any]:
    """Convert C++ algorithm results to pandas-compatible format."""
    # Create vertex results DataFrame
    vertices_df = pd.DataFrame(
        {
            "vertex": original_vertices,
            "community": result["membership"],
            "vertex_weight": result["vertex_weights"],
        }
    )

    # Create community results DataFrame
    community_weights = result["community_weights"]
    communities_df = pd.DataFrame(
        {
            "community": range(len(community_weights)),
            "community_weight": community_weights,
        }
    )

    # Timing and statistics
    stats = {
        "iterations": result["iterations"],
        "passes": result["passes"],
        "total_time": result["time"],
        "affected_vertices": result["affected_vertices"],
    }

    # Algorithm-specific timing
    if "marking_time" in result:  # Louvain
        stats.update(
            {
                "marking_time": result["marking_time"],
                "initialization_time": result["initialization_time"],
                "first_pass_time": result["first_pass_time"],
                "local_move_time": result["local_move_time"],
                "aggregation_time": result["aggregation_time"],
            }
        )

    if "refinement_time" in result:  # Leiden
        stats.update(
            {
                "marking_time": result["marking_time"],
                "initialization_time": result["initialization_time"],
                "first_pass_time": result["first_pass_time"],
                "local_move_time": result["local_move_time"],
                "refinement_time": result["refinement_time"],
                "aggregation_time": result["aggregation_time"],
            }
        )

    return {"vertices": vertices_df, "communities": communities_df, "stats": stats}


def _to_arrow_table(df: pd.DataFrame) -> pa.Table:
    """Convert pandas DataFrame to Apache Arrow Table for zero-copy operations."""
    return pa.Table.from_pandas(df)


def louvain(
    graph: Union[pd.DataFrame, pa.Table],
    directed: bool = True,
    repeat: int = 1,
    resolution: float = 1.0,
    tolerance: float = 1e-2,
    aggregation_tolerance: float = 0.8,
    tolerance_drop: float = 10.0,
    max_iterations: int = 20,
    max_passes: int = 10,
    return_arrow: bool = False,
) -> Dict[str, Union[pd.DataFrame, pa.Table, Dict[str, Any]]]:
    """
    Run Louvain community detection algorithm on a graph.

    Parameters
    ----------
    graph : pd.DataFrame or pa.Table
        Graph data with columns 'source', 'target', and optionally 'weight'.
        Vertices should be integer IDs.
    directed : bool, default True
        Whether to treat the graph as directed.
    repeat : int, default 1
        Number of times to repeat the algorithm.
    resolution : float, default 1.0
        Resolution parameter for modularity optimization.
    tolerance : float, default 1e-2
        Tolerance for convergence.
    aggregation_tolerance : float, default 0.8
        Tolerance for aggregation phase.
    tolerance_drop : float, default 10.0
        Factor to drop tolerance in subsequent iterations.
    max_iterations : int, default 20
        Maximum number of iterations per pass.
    max_passes : int, default 10
        Maximum number of passes.
    return_arrow : bool, default False
        Whether to return Apache Arrow Tables instead of pandas DataFrames.

    Returns
    -------
    dict
        Dictionary containing:
        - 'vertices': DataFrame/Table with vertex community assignments
        - 'communities': DataFrame/Table with community information
        - 'stats': Dictionary with algorithm statistics and timing

    Examples
    --------
    >>> import pandas as pd
    >>> edges = pd.DataFrame({
    ...     'source': [0, 1, 1, 2, 2, 3],
    ...     'target': [1, 2, 0, 3, 1, 2],
    ...     'weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ... })
    >>> result = louvain(edges, directed=False)
    >>> print(result['vertices'])
    >>> print(result['communities'])
    """
    # Import the C++ binding
    try:
        from ._leiden_cpp import louvain as _cpp_louvain
    except ImportError:
        raise ImportError(
            "leiden_communities C++ extension not built. Run: pip install -e ."
        )

    # Convert Arrow Table to DataFrame if needed
    if isinstance(graph, pa.Table):
        graph = graph.to_pandas()

    # Validate input
    _validate_graph_dataframe(graph)

    # Prepare arrays for C++ binding
    sources, targets, weights, original_vertices, vertex_map = _prepare_graph_arrays(
        graph
    )

    # Call C++ implementation
    result = _cpp_louvain(
        sources=sources,
        targets=targets,
        weights=weights,
        directed=directed,
        repeat=repeat,
        resolution=resolution,
        tolerance=tolerance,
        aggregation_tolerance=aggregation_tolerance,
        tolerance_drop=tolerance_drop,
        max_iterations=max_iterations,
        max_passes=max_passes,
    )

    # Convert results to DataFrame format
    formatted_result = _convert_result_to_dataframe(result, original_vertices)

    # Convert to Arrow Tables if requested
    if return_arrow:
        formatted_result["vertices"] = _to_arrow_table(formatted_result["vertices"])
        formatted_result["communities"] = _to_arrow_table(
            formatted_result["communities"]
        )

    return formatted_result


def leiden(
    graph: Union[pd.DataFrame, pa.Table],
    directed: bool = True,
    repeat: int = 1,
    resolution: float = 1.0,
    tolerance: float = 1e-2,
    aggregation_tolerance: float = 0.8,
    tolerance_drop: float = 10.0,
    max_iterations: int = 20,
    max_passes: int = 10,
    return_arrow: bool = False,
) -> Dict[str, Union[pd.DataFrame, pa.Table, Dict[str, Any]]]:
    """
    Run Leiden community detection algorithm on a graph.

    Parameters
    ----------
    graph : pd.DataFrame or pa.Table
        Graph data with columns 'source', 'target', and optionally 'weight'.
        Vertices should be integer IDs.
    directed : bool, default True
        Whether to treat the graph as directed.
    repeat : int, default 1
        Number of times to repeat the algorithm.
    resolution : float, default 1.0
        Resolution parameter for modularity optimization.
    tolerance : float, default 1e-2
        Tolerance for convergence.
    aggregation_tolerance : float, default 0.8
        Tolerance for aggregation phase.
    tolerance_drop : float, default 10.0
        Factor to drop tolerance in subsequent iterations.
    max_iterations : int, default 20
        Maximum number of iterations per pass.
    max_passes : int, default 10
        Maximum number of passes.
    return_arrow : bool, default False
        Whether to return Apache Arrow Tables instead of pandas DataFrames.

    Returns
    -------
    dict
        Dictionary containing:
        - 'vertices': DataFrame/Table with vertex community assignments
        - 'communities': DataFrame/Table with community information
        - 'stats': Dictionary with algorithm statistics and timing

    Examples
    --------
    >>> import pandas as pd
    >>> edges = pd.DataFrame({
    ...     'source': [0, 1, 1, 2, 2, 3],
    ...     'target': [1, 2, 0, 3, 1, 2],
    ...     'weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ... })
    >>> result = leiden(edges, directed=False)
    >>> print(result['vertices'])
    >>> print(result['communities'])
    """
    # Import the C++ binding
    try:
        from ._leiden_cpp import leiden as _cpp_leiden
    except ImportError:
        raise ImportError(
            "leiden_communities C++ extension not built. Run: pip install -e ."
        )

    # Convert Arrow Table to DataFrame if needed
    if isinstance(graph, pa.Table):
        graph = graph.to_pandas()

    # Validate input
    _validate_graph_dataframe(graph)

    # Prepare arrays for C++ binding
    sources, targets, weights, original_vertices, vertex_map = _prepare_graph_arrays(
        graph
    )

    # Call C++ implementation
    result = _cpp_leiden(
        sources=sources,
        targets=targets,
        weights=weights,
        directed=directed,
        repeat=repeat,
        resolution=resolution,
        tolerance=tolerance,
        aggregation_tolerance=aggregation_tolerance,
        tolerance_drop=tolerance_drop,
        max_iterations=max_iterations,
        max_passes=max_passes,
    )

    # Convert results to DataFrame format
    formatted_result = _convert_result_to_dataframe(result, original_vertices)

    # Convert to Arrow Tables if requested
    if return_arrow:
        formatted_result["vertices"] = _to_arrow_table(formatted_result["vertices"])
        formatted_result["communities"] = _to_arrow_table(
            formatted_result["communities"]
        )

    return formatted_result


def from_edge_list(
    edges: Union[list, np.ndarray, pd.DataFrame],
    weights: Optional[Union[list, np.ndarray, pd.Series]] = None,
    directed: bool = True,
) -> pd.DataFrame:
    """
    Create a graph DataFrame from an edge list.

    Parameters
    ----------
    edges : list, array, or DataFrame
        Edge list as pairs of vertex IDs.
    weights : list, array, or Series, optional
        Edge weights. If None, all edges have weight 1.0.
    directed : bool, default True
        Whether the graph is directed.

    Returns
    -------
    pd.DataFrame
        Graph DataFrame with 'source', 'target', and 'weight' columns.
    """
    if isinstance(edges, pd.DataFrame):
        return edges

    edges = np.array(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Edges must be a 2D array with shape (n_edges, 2)")

    df = pd.DataFrame({"source": edges[:, 0], "target": edges[:, 1]})

    if weights is not None:
        df["weight"] = weights
    else:
        df["weight"] = 1.0

    return df


def to_arrow_table(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert algorithm results to Apache Arrow format.

    Parameters
    ----------
    result : dict
        Result dictionary from louvain() or leiden() with return_arrow=False.

    Returns
    -------
    dict
        Result dictionary with DataFrames converted to Arrow Tables.
    """
    arrow_result = result.copy()

    if "vertices" in result and isinstance(result["vertices"], pd.DataFrame):
        arrow_result["vertices"] = _to_arrow_table(result["vertices"])

    if "communities" in result and isinstance(result["communities"], pd.DataFrame):
        arrow_result["communities"] = _to_arrow_table(result["communities"])

    return arrow_result


def to_pandas(
    result: Union[Dict[str, Any], pa.Table]
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Convert Arrow Table results to pandas DataFrame format.

    Parameters
    ----------
    result : dict or pa.Table
        Result dictionary from louvain() or leiden() with return_arrow=True,
        or a single Arrow Table.

    Returns
    -------
    dict or pd.DataFrame
        Result dictionary with Arrow Tables converted to DataFrames,
        or a single DataFrame if input was a Table.
    """
    # Handle single Arrow Table
    if isinstance(result, pa.Table):
        return result.to_pandas()

    # Handle result dictionary
    pandas_result = result.copy()

    if "vertices" in result and isinstance(result["vertices"], pa.Table):
        pandas_result["vertices"] = result["vertices"].to_pandas()

    if "communities" in result and isinstance(result["communities"], pa.Table):
        pandas_result["communities"] = result["communities"].to_pandas()

    return pandas_result
