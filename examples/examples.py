#!/usr/bin/env python3
"""
Examples demonstrating the leiden_communities package usage with pandas and Arrow.
"""

import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow as pa

# Import our community detection algorithms
import leiden_communities as lc


def create_example_graph() -> pd.DataFrame:
    """Create a simple example graph for testing."""
    # Create a small graph with two clear communities
    edges = [
        # Community 1: nodes 0-2
        (0, 1, 1.0),
        (1, 0, 1.0),  # Bidirectional edges
        (1, 2, 1.0),
        (2, 1, 1.0),
        (0, 2, 0.5),
        (2, 0, 0.5),
        # Community 2: nodes 3-5
        (3, 4, 1.0),
        (4, 3, 1.0),
        (4, 5, 1.0),
        (5, 4, 1.0),
        (3, 5, 0.5),
        (5, 3, 0.5),
        # Weak inter-community connections
        (2, 3, 0.1),
        (3, 2, 0.1),
    ]

    df = pd.DataFrame(edges, columns=["source", "target", "weight"])
    return df


def create_larger_graph(
    n_communities: int = 5, nodes_per_community: int = 20
) -> pd.DataFrame:
    """Create a larger synthetic graph with clear community structure."""
    edges = []

    for comm in range(n_communities):
        start_node = comm * nodes_per_community
        end_node = start_node + nodes_per_community

        # Dense intra-community connections
        for i in range(start_node, end_node):
            for j in range(i + 1, end_node):
                if np.random.random() < 0.3:  # 30% chance of connection
                    weight = np.random.uniform(0.5, 1.5)
                    edges.extend([(i, j, weight), (j, i, weight)])

        # Sparse inter-community connections
        if comm < n_communities - 1:
            next_start = (comm + 1) * nodes_per_community
            next_end = next_start + nodes_per_community

            for _ in range(2):  # Only a few connections between communities
                i = np.random.randint(start_node, end_node)
                j = np.random.randint(next_start, next_end)
                weight = np.random.uniform(0.1, 0.3)
                edges.extend([(i, j, weight), (j, i, weight)])

    df = pd.DataFrame(edges, columns=["source", "target", "weight"])
    return df


def print_results(result: Dict[str, Any], algorithm_name: str) -> None:
    """Print algorithm results in a readable format."""
    print(f"\n=== {algorithm_name} Results ===")

    vertices_df = result["vertices"]
    communities_df = result["communities"]
    stats = result["stats"]

    print(f"Number of vertices: {len(vertices_df)}")
    print(f"Number of communities found: {len(communities_df)}")
    print(f"Iterations: {stats['iterations']}")
    print(f"Passes: {stats['passes']}")
    print(f"Total time: {stats['total_time']:.4f} seconds")
    print(f"Affected vertices: {stats['affected_vertices']}")

    # Show community assignments
    print("\nCommunity assignments:")
    community_counts = vertices_df["community"].value_counts().sort_index()
    for comm_id, count in community_counts.items():
        print(f"  Community {comm_id}: {count} vertices")

    # Show timing breakdown if available
    if "marking_time" in stats:
        print("\nTiming breakdown:")
        print(f"  Marking time: {stats['marking_time']:.4f}s")
        print(f"  Initialization time: {stats['initialization_time']:.4f}s")
        print(f"  First pass time: {stats['first_pass_time']:.4f}s")
        print(f"  Local move time: {stats['local_move_time']:.4f}s")
        print(f"  Aggregation time: {stats['aggregation_time']:.4f}s")
        if "refinement_time" in stats:
            print(f"  Refinement time: {stats['refinement_time']:.4f}s")


def example_basic_usage():
    """Demonstrate basic usage with a simple graph."""
    print("=== Basic Usage Example ===")

    # Create example graph
    graph_df = create_example_graph()
    print(f"Created graph with {len(graph_df)} edges")
    print(graph_df.head())

    # Run Louvain algorithm
    louvain_result = lc.louvain(graph_df, directed=False)
    print_results(louvain_result, "Louvain")

    # Run Leiden algorithm
    leiden_result = lc.leiden(graph_df, directed=False)
    print_results(leiden_result, "Leiden")


def example_from_edge_list():
    """Demonstrate creating graph from edge list."""
    print("\n=== Edge List Example ===")

    # Create from simple edge list
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (1, 4)]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]

    graph_df = lc.from_edge_list(edges, weights=weights, directed=False)
    print("Graph DataFrame:")
    print(graph_df)

    result = lc.leiden(graph_df, directed=False)
    print_results(result, "Leiden from Edge List")


def example_arrow_integration():
    """Demonstrate Apache Arrow integration."""
    print("\n=== Apache Arrow Integration Example ===")

    # Create graph
    graph_df = create_example_graph()

    # Convert to Arrow Table
    graph_arrow = pa.Table.from_pandas(graph_df)
    print(f"Created Arrow Table with {graph_arrow.num_rows} rows")

    # Run algorithm with Arrow input and output
    result_arrow = lc.leiden(graph_arrow, directed=False, return_arrow=True)

    print("Result types:")
    print(f"  Vertices: {type(result_arrow['vertices'])}")
    print(f"  Communities: {type(result_arrow['communities'])}")

    # Convert back to pandas for display
    result_pandas = lc.to_pandas(result_arrow)
    print_results(result_pandas, "Leiden with Arrow")

    # Demonstrate zero-copy efficiency
    vertices_arrow = result_arrow["vertices"]
    print("\nArrow Table schema:")
    print(vertices_arrow.schema)


def example_parameter_tuning():
    """Demonstrate parameter tuning for different scenarios."""
    print("\n=== Parameter Tuning Example ===")

    graph_df = create_larger_graph(n_communities=3, nodes_per_community=15)
    print(f"Created larger graph with {len(graph_df)} edges")

    # Default parameters
    print("\n--- Default Parameters ---")
    result_default = lc.leiden(graph_df, directed=False)
    print_results(result_default, "Leiden (Default)")

    # Higher resolution (more communities)
    print("\n--- Higher Resolution ---")
    result_high_res = lc.leiden(graph_df, directed=False, resolution=2.0)
    print_results(result_high_res, "Leiden (High Resolution)")

    # Lower resolution (fewer communities)
    print("\n--- Lower Resolution ---")
    result_low_res = lc.leiden(graph_df, directed=False, resolution=0.5)
    print_results(result_low_res, "Leiden (Low Resolution)")


def example_performance_comparison():
    """Compare performance between Louvain and Leiden algorithms."""
    print("\n=== Performance Comparison ===")

    # Create a larger graph for meaningful timing
    graph_df = create_larger_graph(n_communities=8, nodes_per_community=50)
    print(f"Created performance test graph with {len(graph_df)} edges")

    # Run Louvain
    start_time = time.time()
    louvain_result = lc.louvain(graph_df, directed=False, repeat=5)
    louvain_time = time.time() - start_time

    # Run Leiden
    start_time = time.time()
    leiden_result = lc.leiden(graph_df, directed=False, repeat=5)
    leiden_time = time.time() - start_time

    print("\nPerformance Results:")
    print(
        f"Louvain: {louvain_time:.4f}s total, found {len(louvain_result['communities'])} communities"
    )
    print(
        f"Leiden:  {leiden_time:.4f}s total, found {len(leiden_result['communities'])} communities"
    )
    print(
        f"Speedup: {louvain_time/leiden_time:.2f}x {'(Leiden faster)' if leiden_time < louvain_time else '(Louvain faster)'}"
    )


def example_error_handling():
    """Demonstrate error handling and validation."""
    print("\n=== Error Handling Example ===")

    try:
        # Missing required columns
        bad_df = pd.DataFrame({"src": [0, 1], "dst": [1, 2]})
        lc.louvain(bad_df)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=["source", "target"])
        lc.leiden(empty_df)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        # Negative vertex IDs
        negative_df = pd.DataFrame({"source": [-1, 0], "target": [0, 1]})
        lc.louvain(negative_df)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("Error handling working correctly!")


def main():
    """Run all examples."""
    print("Running leiden_communities examples...")

    try:
        example_basic_usage()
        example_from_edge_list()
        example_arrow_integration()
        example_parameter_tuning()
        example_performance_comparison()
        example_error_handling()

        print("\n=== All Examples Completed Successfully! ===")

    except ImportError as e:
        print(f"Error: {e}")
        print("Please build and install the package first:")
        print("  pip install -e .")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
