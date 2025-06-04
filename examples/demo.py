#!/usr/bin/env python3
"""
Demonstration of the leiden_communities Python bindings.

This script shows the main functionality of the high-performance Leiden and Louvain
community detection algorithms with pandas DataFrame and Apache Arrow support.
"""

import numpy as np
import pandas as pd
import pyarrow as pa

from leiden_communities import from_edge_list, leiden, louvain, to_pandas


def main():
    print("=" * 60)
    print("LEIDEN COMMUNITIES PYTHON BINDINGS DEMONSTRATION")
    print("=" * 60)

    # Create a sample graph
    print("\n1. Creating sample graph data...")
    edges = pd.DataFrame(
        {
            "source": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3],
            "target": [1, 2, 0, 3, 1, 4, 2, 5, 3, 0, 2, 5],
            "weight": [1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 0.3, 0.8],
        }
    )
    print(
        f"Created graph with {len(edges)} edges and {len(pd.concat([edges['source'], edges['target']]).unique())} vertices"
    )
    print(edges)

    # Test Louvain algorithm
    print("\n2. Testing Louvain community detection...")
    louvain_result = louvain(edges, directed=False, resolution=1.0)
    print(f"✓ Found {len(louvain_result['communities'])} communities")
    print(f"✓ Algorithm stats: {louvain_result['stats']}")
    print("\nVertex assignments:")
    print(louvain_result["vertices"].head())
    print("\nCommunity weights:")
    print(louvain_result["communities"].head())

    # Test Leiden algorithm
    print("\n3. Testing Leiden community detection...")
    leiden_result = leiden(edges, directed=False, resolution=1.0)
    print(f"✓ Found {len(leiden_result['communities'])} communities")
    print(f"✓ Algorithm stats: {leiden_result['stats']}")
    print("\nVertex assignments:")
    print(leiden_result["vertices"].head())

    # Test Apache Arrow integration
    print("\n4. Testing Apache Arrow integration...")
    edges_arrow = pa.Table.from_pandas(edges)
    print(f"Converted DataFrame to Arrow Table: {edges_arrow.shape}")

    arrow_result = louvain(edges_arrow, directed=False, return_arrow=True)
    print("✓ Results returned as Arrow Tables")
    print(f"  - Vertices: {type(arrow_result['vertices'])}")
    print(f"  - Communities: {type(arrow_result['communities'])}")

    # Convert back to pandas
    vertices_df = to_pandas(arrow_result["vertices"])
    communities_df = to_pandas(arrow_result["communities"])
    print(f"✓ Converted back to pandas: {vertices_df.shape}, {communities_df.shape}")

    # Test edge list utility
    print("\n5. Testing utility functions...")
    edge_list = [(0, 1), (1, 2), (2, 0)]  # List of 2-tuples
    edge_weights = [1.0, 1.0, 0.5]  # Separate weights
    df_from_list = from_edge_list(edge_list, weights=edge_weights)
    print(f"✓ Created DataFrame from edge list: {df_from_list.shape}")
    print(df_from_list)

    # Performance test with larger graph
    print("\n6. Performance test with larger random graph...")
    np.random.seed(42)
    n_vertices = 500
    n_edges = 2000

    large_edges = pd.DataFrame(
        {
            "source": np.random.randint(0, n_vertices, n_edges),
            "target": np.random.randint(0, n_vertices, n_edges),
            "weight": np.random.uniform(0.1, 1.0, n_edges),
        }
    )
    # Remove self-loops
    large_edges = large_edges[
        large_edges["source"] != large_edges["target"]
    ].drop_duplicates()

    print(
        f"Created random graph with {len(large_edges)} edges, "
        f"{len(pd.concat([large_edges['source'], large_edges['target']]).unique())} vertices"
    )

    import time

    start_time = time.time()
    large_louvain = louvain(large_edges, directed=False)
    louvain_time = time.time() - start_time

    start_time = time.time()
    large_leiden = leiden(large_edges, directed=False)
    leiden_time = time.time() - start_time

    print(
        f"✓ Louvain: {louvain_time:.3f}s, {len(large_louvain['communities'])} communities"
    )
    print(
        f"✓ Leiden: {leiden_time:.3f}s, {len(large_leiden['communities'])} communities"
    )

    print("\n" + "=" * 60)
    print("🎉 DEMONSTRATION COMPLETE - All functionality working!")
    print("=" * 60)


if __name__ == "__main__":
    main()
