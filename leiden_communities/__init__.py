"""
Python bindings for high-performance Leiden and Louvain community detection algorithms.

This module provides efficient community detection algorithms with pandas DataFrame
and Apache Arrow support for optimal memory usage and performance.
"""

import os

try:
    # Import high-level wrapper functions
    # Also make low-level C++ functions available
    from . import _leiden_cpp as _cpp
    from .wrapper import from_edge_list, leiden, louvain, to_arrow_table, to_pandas

    # Array-based modularity functions
    def modularity(
        sources, targets, membership, weights=None, directed=True, resolution=1.0
    ):
        """Calculate modularity from edge arrays and community membership.

        Parameters
        ----------
        sources : array-like
            Source vertex IDs for each edge
        targets : array-like
            Target vertex IDs for each edge
        membership : array-like
            Community membership for each vertex
        weights : array-like, optional
            Edge weights. If None, all edges have weight 1.0
        directed : bool, default True
            Whether the graph is directed
        resolution : float, default 1.0
            Resolution parameter for modularity calculation

        Returns
        -------
        float
            Modularity value
        """
        import numpy as np

        sources = np.asarray(sources, dtype=np.uint32)
        targets = np.asarray(targets, dtype=np.uint32)
        membership = np.asarray(membership, dtype=np.uint32)

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
        else:
            weights = np.array([], dtype=np.float64)

        return _cpp.modularity(
            sources, targets, membership, weights, directed, resolution
        )

    # OpenMP thread control functions
    def get_max_threads():
        """Get the maximum number of OpenMP threads available."""
        return _cpp.get_max_threads()

    def get_num_threads():
        """Get the current number of OpenMP threads in use."""
        return _cpp.get_num_threads()

    def set_num_threads(num_threads=None):
        """Set the number of OpenMP threads to use.

        Parameters
        ----------
        num_threads : int, optional
            Number of threads to use. If None, uses all available CPUs.
            If 0 or negative, uses all available CPUs.
        """
        if num_threads is None:
            num_threads = os.cpu_count()
        elif num_threads <= 0:
            num_threads = os.cpu_count()

        _cpp.set_num_threads(int(num_threads))

    # Automatically set threads to use all available CPUs unless OMP_NUM_THREADS is set
    if "OMP_NUM_THREADS" not in os.environ:
        set_num_threads()

except ImportError as e:
    # Module not built yet - provide informative error messages
    error_msg = str(e)

    def louvain(*args, **kwargs):
        raise ImportError(
            f"leiden_communities extension not built. Run: pip install -e . Error: {error_msg}"
        )

    def leiden(*args, **kwargs):
        raise ImportError(
            f"leiden_communities extension not built. Run: pip install -e . Error: {error_msg}"
        )

    def from_edge_list(*args, **kwargs):
        raise ImportError(
            f"leiden_communities extension not built. Run: pip install -e . Error: {error_msg}"
        )

    def to_arrow_table(*args, **kwargs):
        raise ImportError(
            f"leiden_communities extension not built. Run: pip install -e . Error: {error_msg}"
        )

    def to_pandas(*args, **kwargs):
        raise ImportError(
            f"leiden_communities extension not built. Run: pip install -e . Error: {error_msg}"
        )

    def modularity(*args, **kwargs):
        raise ImportError(
            f"leiden_communities extension not built. Run: pip install -e . Error: {error_msg}"
        )

    def get_max_threads(*args, **kwargs):
        raise ImportError(
            f"leiden_communities extension not built. Run: pip install -e . Error: {error_msg}"
        )

    def get_num_threads(*args, **kwargs):
        raise ImportError(
            f"leiden_communities extension not built. Run: pip install -e . Error: {error_msg}"
        )

    def set_num_threads(*args, **kwargs):
        raise ImportError(
            f"leiden_communities extension not built. Run: pip install -e . Error: {error_msg}"
        )


__version__ = "0.1.0"
__all__ = [
    "louvain",
    "leiden",
    "from_edge_list",
    "to_arrow_table",
    "to_pandas",
    "modularity",
    "get_max_threads",
    "get_num_threads",
    "set_num_threads",
]
