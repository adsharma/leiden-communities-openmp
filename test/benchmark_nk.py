#!/usr/bin/env python3

import argparse
import time
import tempfile
import os

import numpy as np
import pandas as pd
import networkit as nk

import leiden_communities as lc


def create_test_graph(n_vertices=1000, n_edges=5000):
    """Create a random test graph"""
    np.random.seed(42)
    sources = np.random.randint(0, n_vertices, n_edges, dtype=np.uint32)
    targets = np.random.randint(0, n_vertices, n_edges, dtype=np.uint32)
    weights = np.random.random(n_edges) * 10

    # Remove self-loops
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]
    weights = weights[mask]

    return sources, targets, weights


def read_graph_from_csv(csv_file):
    """Read graph from CSV file
    Expected CSV format: source,target,weight (with header)
    or: node1,node2,weight
    """
    try:
        df = pd.read_csv(csv_file)

        # Try different common column name patterns
        if "source" in df.columns and "target" in df.columns:
            sources = df["source"].values
            targets = df["target"].values
        elif "node1" in df.columns and "node2" in df.columns:
            sources = df["node1"].values
            targets = df["node2"].values
        elif "from" in df.columns and "to" in df.columns:
            sources = df["from"].values
            targets = df["to"].values
        else:
            # Assume first two columns are source and target
            sources = df.iloc[:, 0].values
            targets = df.iloc[:, 1].values

        # Handle weights
        if "weight" in df.columns:
            weights = df["weight"].values
        elif "w" in df.columns:
            weights = df["w"].values
        elif df.shape[1] >= 3:
            # Assume third column is weight
            weights = df.iloc[:, 2].values
        else:
            # Default to weight 1.0 for all edges
            weights = np.ones(len(sources))

        # Convert to appropriate dtypes
        sources = sources.astype(np.uint32)
        targets = targets.astype(np.uint32)
        weights = weights.astype(np.float64)

        # Remove self-loops
        mask = sources != targets
        sources = sources[mask]
        targets = targets[mask]
        weights = weights[mask]

        return sources, targets, weights

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Expected CSV format: source,target,weight (with header)")
        print("Alternative formats supported: node1,node2,weight or from,to,weight")
        raise


def calculate_modularity_from_result(sources, targets, weights, result):
    """
    Helper function to calculate modularity from algorithm result.
    Uses lc.modularity directly with the membership array from the result.
    """
    # Extract membership array from result - handle both high-level and low-level formats
    if isinstance(result, dict):
        if "vertices" in result:
            # High-level wrapper format (DataFrame result)
            vertices_df = result["vertices"]
            vertex_ids = vertices_df["vertex"].values.astype(np.uint32)
            communities = vertices_df["community"].values.astype(np.uint32)
            max_vertex = max(max(sources), max(targets))
            membership = np.zeros(max_vertex + 1, dtype=np.uint32)
            membership[vertex_ids] = communities
        elif "membership" in result:
            # Low-level format (direct membership array)
            membership = result["membership"].astype(np.uint32)
        else:
            raise ValueError(f"Unexpected result format: {type(result)}")
    else:
        raise ValueError(f"Unexpected result format: {type(result)}")

    # Calculate modularity using the exposed OpenMP function
    return lc.modularity(sources, targets, membership, weights, directed=False)


def calculate_modularity_networkit(G, partition):
    """Calculate modularity using NetworkIt"""
    modularity = nk.community.Modularity()
    return modularity.getQuality(partition, G)


def create_networkit_graph(sources, targets, weights):
    """Create a NetworkIt graph from edge data"""
    # Map vertices to continuous range starting from 0
    unique_vertices = np.unique(np.concatenate([sources, targets]))
    vertex_map = {v: i for i, v in enumerate(unique_vertices)}
    n_vertices = len(unique_vertices)

    # Create NetworkIt graph
    G = nk.Graph(n_vertices, weighted=True, directed=False)

    # Add edges with remapped vertex IDs
    for s, t, w in zip(sources, targets, weights):
        G.addEdge(vertex_map[s], vertex_map[t], w)

    return G, vertex_map


def create_temp_csv(sources, targets, weights):
    """Create a temporary CSV file for NetworkIt EdgeListReader"""
    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".csv", text=True)

    try:
        with os.fdopen(temp_fd, "w") as f:
            f.write("source,target,weight\n")
            for s, t, w in zip(sources, targets, weights):
                f.write(f"{s},{t},{w}\n")
        return temp_path
    except:
        os.close(temp_fd)
        raise


def benchmark_algorithms(graph_data=None, num_iterations=3):
    if graph_data is None:
        print("Creating test graph...")
        sources, targets, weights = create_test_graph(n_vertices=2000, n_edges=10000)
    else:
        print("Using provided graph data...")
        sources, targets, weights = graph_data

    print(
        f"Graph: {len(sources)} edges, {max(max(sources), max(targets)) + 1} vertices"
    )
    print()

    # Create graphs for both libraries
    print("Preparing graphs...")

    # For leiden-communities-openmp
    graph_df = pd.DataFrame({"source": sources, "target": targets, "weight": weights})

    # For NetworkIt - create graph directly
    nk_graph, vertex_map = create_networkit_graph(sources, targets, weights)

    print(
        f"NetworkIt graph: {nk_graph.numberOfEdges()} edges, {nk_graph.numberOfNodes()} nodes"
    )
    print()

    # ==========================================
    # Benchmark leiden-communities-openmp Leiden
    # ==========================================
    print("Benchmarking leiden-communities-openmp Leiden...")
    times = []
    lc_leiden_result = None
    for i in range(num_iterations):
        start_time = time.perf_counter()
        result = lc.leiden(graph_df, directed=False)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f}s")
        if i == 0:  # Save first result for modularity calculation
            lc_leiden_result = result

    avg_lc_leiden = (
        np.mean(times[1:]) if len(times) > 1 else times[0]
    )  # Exclude first run (warmup) if multiple runs
    print(f"  Average time (excluding warmup): {avg_lc_leiden:.4f}s")

    # Calculate modularity
    lc_leiden_modularity = calculate_modularity_from_result(
        sources, targets, weights, lc_leiden_result
    )
    lc_leiden_communities = len(lc_leiden_result["communities"])
    print(f"  Communities found: {lc_leiden_communities:.6f}")
    print(f"  Modularity: {lc_leiden_modularity:.6f}")
    print()

    # ==========================================
    # Benchmark NetworkIt Leiden
    # ==========================================
    print("Benchmarking NetworkIt Leiden...")
    times = []
    nk_leiden_result = None
    for i in range(num_iterations):
        start_time = time.perf_counter()
        plm = nk.community.ParallelLeiden(nk_graph)
        plm.run()
        partition = plm.getPartition()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f}s")
        if i == 0:  # Save first result for modularity calculation
            nk_leiden_result = partition

    avg_nk_leiden = (
        np.mean(times[1:]) if len(times) > 1 else times[0]
    )  # Exclude first run (warmup) if multiple runs
    print(f"  Average time (excluding warmup): {avg_nk_leiden:.4f}s")

    # Calculate modularity
    nk_leiden_modularity = calculate_modularity_networkit(nk_graph, nk_leiden_result)
    nk_leiden_communities = nk_leiden_result.numberOfSubsets()
    print(f"  Communities found: {nk_leiden_communities:.6f}")
    print(f"  Modularity: {nk_leiden_modularity:.6f}")
    print()

    # === RESULTS SUMMARY ===
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("Algorithm Performance:")
    print(f"  leiden-communities-openmp Leiden:  {avg_lc_leiden:.4f}s")
    print(f"  NetworkIt Leiden:                  {avg_nk_leiden:.4f}s")
    print()

    print("Performance Comparison:")
    speedup_vs_nk_leiden = avg_nk_leiden / avg_lc_leiden
    print(
        f"  leiden-communities-openmp vs NetworkIt Leiden:  {speedup_vs_nk_leiden:.2f}x"
    )
    print()

    print("Community Detection Results:")
    print(
        f"  leiden-communities-openmp Leiden:  {lc_leiden_communities} communities, modularity {lc_leiden_modularity:.6f}"
    )
    print(
        f"  NetworkIt Leiden:                  {nk_leiden_communities} communities, modularity {nk_leiden_modularity}"
    )
    print()

    print("Quality Comparison:")
    print(
        f"  Modularity difference (LC vs NK Leiden):  {abs(lc_leiden_modularity - nk_leiden_modularity):.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark of leiden-communities-openmp vs NetworkIt algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This benchmark compares the performance of leiden-communities-openmp with NetworkIt's
Leiden and Louvain implementations.

CSV File Format:
  The CSV file should contain graph edges with the following supported formats:
  1. source,target,weight (with header)
  2. node1,node2,weight (with header)
  3. from,to,weight (with header)
  4. First two columns as source/target, third column (optional) as weight

  If no weight column is provided, all edges will have weight 1.0.
  Self-loops are automatically removed.

Example usage:
  python benchmark_vs_networkit.py                    # Test with random graph
  python benchmark_vs_networkit.py --csv data.csv     # Test with your graph
  python benchmark_vs_networkit.py --iters 5          # Run 5 iterations per algorithm
        """,
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file containing graph edges. If not provided, a random test graph will be generated.",
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Number of iterations to run for each algorithm (default: 3).",
    )

    args = parser.parse_args()

    graph_data = None
    if args.csv:
        try:
            graph_data = read_graph_from_csv(args.csv)
            print(f"Successfully loaded graph from {args.csv}")
        except Exception as e:
            print(f"Failed to load graph from {args.csv}: {e}")
            print("Falling back to generated test graph...")
            graph_data = None

    benchmark_algorithms(graph_data, args.iters)
