#!/usr/bin/env python3
"""
Test script for leiden_communities package.

This script tests both the high-level pandas/Arrow interfaces and the
low-level C++ bindings to ensure everything is working correctly.
"""

import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa


def test_basic_import():
    """Test that the package can be imported."""
    print("Testing basic import...")
    try:
        import leiden_communities

        print(
            f"✓ Successfully imported leiden_communities v{leiden_communities.__version__}"
        )
    except ImportError as e:
        print(f"✗ Failed to import leiden_communities: {e}")
        raise


def test_low_level_bindings():
    """Test the low-level C++ bindings directly."""
    print("\nTesting low-level C++ bindings...")
    try:
        from leiden_communities._leiden_cpp import leiden as cpp_leiden
        from leiden_communities._leiden_cpp import louvain as cpp_louvain

        # Create simple test graph: 0-1-2-3 with some cross connections
        sources = np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32)
        targets = np.array([1, 2, 3, 0, 1, 2], dtype=np.uint32)
        weights = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 1.0], dtype=np.float64)

        # Test Louvain
        print("  Testing Louvain...")
        louvain_result = cpp_louvain(sources, targets, weights, directed=False)
        print(f"    ✓ Louvain completed in {louvain_result['time']:.4f}s")
        print(f"    ✓ Found communities: {len(set(louvain_result['membership']))}")
        print(
            f"    ✓ Iterations: {louvain_result['iterations']}, Passes: {louvain_result['passes']}"
        )

        # Test Leiden
        print("  Testing Leiden...")
        leiden_result = cpp_leiden(sources, targets, weights, directed=False)
        print(f"    ✓ Leiden completed in {leiden_result['time']:.4f}s")
        print(f"    ✓ Found communities: {len(set(leiden_result['membership']))}")
        print(
            f"    ✓ Iterations: {leiden_result['iterations']}, Passes: {leiden_result['passes']}"
        )

    except Exception as e:
        print(f"    ✗ Low-level binding test failed: {e}")
        raise


def test_high_level_pandas():
    """Test the high-level pandas DataFrame interface."""
    print("\nTesting high-level pandas interface...")
    try:
        from leiden_communities import leiden, louvain

        # Create test graph as DataFrame
        edges_df = pd.DataFrame(
            {
                "source": [0, 1, 2, 3, 1, 2, 0, 3],
                "target": [1, 2, 3, 0, 0, 1, 3, 1],
                "weight": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.3, 0.3],
            }
        )

        print(f"  Created test graph with {len(edges_df)} edges")

        # Test Louvain
        print("  Testing Louvain with pandas...")
        louvain_result = louvain(edges_df, directed=False, resolution=1.0)
        vertices_df = louvain_result["vertices"]
        communities_df = louvain_result["communities"]
        stats = louvain_result["stats"]

        print(f"    ✓ Louvain completed in {stats['total_time']:.4f}s")
        print(f"    ✓ Found {len(communities_df)} communities")
        print(f"    ✓ Processed {len(vertices_df)} vertices")
        print(
            f"    ✓ Result structure: vertices={type(vertices_df).__name__}, communities={type(communities_df).__name__}"
        )

        # Test Leiden
        print("  Testing Leiden with pandas...")
        leiden_result = leiden(edges_df, directed=False, resolution=1.0)
        vertices_df = leiden_result["vertices"]
        communities_df = leiden_result["communities"]
        stats = leiden_result["stats"]

        print(f"    ✓ Leiden completed in {stats['total_time']:.4f}s")
        print(f"    ✓ Found {len(communities_df)} communities")
        print(f"    ✓ Processed {len(vertices_df)} vertices")
        print(f"    ✓ Has refinement_time: {'refinement_time' in stats}")

    except Exception as e:
        print(f"    ✗ Pandas interface test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_arrow_interface():
    """Test the Apache Arrow interface."""
    print("\nTesting Apache Arrow interface...")
    try:
        from leiden_communities import louvain, to_pandas

        # Create test graph as DataFrame
        edges_df = pd.DataFrame(
            {
                "source": [0, 1, 2, 3, 1, 2],
                "target": [1, 2, 3, 0, 0, 1],
                "weight": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
            }
        )

        # Convert to Arrow Table
        edges_arrow = pa.Table.from_pandas(edges_df)
        print(f"  Created Arrow table with {len(edges_arrow)} rows")

        # Test with Arrow input and Arrow output
        print("  Testing Louvain with Arrow input/output...")
        louvain_result = louvain(edges_arrow, directed=False, return_arrow=True)

        vertices_table = louvain_result["vertices"]
        communities_table = louvain_result["communities"]
        stats = louvain_result["stats"]

        print(f"    ✓ Louvain completed in {stats['total_time']:.4f}s")
        print(
            f"    ✓ Result types: vertices={type(vertices_table).__name__}, communities={type(communities_table).__name__}"
        )
        print(f"    ✓ Arrow vertices columns: {vertices_table.column_names}")
        print(f"    ✓ Arrow communities columns: {communities_table.column_names}")

        # Test conversion back to pandas
        pandas_result = to_pandas(louvain_result)
        print(
            f"    ✓ Converted back to pandas: {type(pandas_result['vertices']).__name__}"
        )

    except Exception as e:
        print(f"    ✗ Arrow interface test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_edge_cases():
    """Test various edge cases and error conditions."""
    print("\nTesting edge cases...")
    try:
        from leiden_communities import from_edge_list, louvain

        # Test from_edge_list utility
        edges_list = [(0, 1), (1, 2), (2, 0)]
        edges_df = from_edge_list(edges_list)
        print(
            f"  ✓ from_edge_list created DataFrame with columns: {list(edges_df.columns)}"
        )

        # Test with no weights
        edges_no_weights = pd.DataFrame({"source": [0, 1, 2], "target": [1, 2, 0]})
        louvain(edges_no_weights, directed=False)
        print("  ✓ Louvain works without weight column")

        # Test error handling
        try:
            bad_df = pd.DataFrame({"src": [0, 1], "dst": [1, 2]})  # Wrong column names
            louvain(bad_df)
            print("  ✗ Should have failed with wrong column names")
            raise AssertionError("Should have failed with wrong column names")
        except ValueError:
            print("  ✓ Correctly rejected DataFrame with wrong column names")

    except Exception as e:
        print(f"    ✗ Edge cases test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_performance():
    """Test performance with a larger graph."""
    print("\nTesting performance with larger graph...")
    try:
        from leiden_communities import leiden, louvain

        # Create a larger random graph
        np.random.seed(42)
        n_vertices = 1000
        n_edges = 5000

        sources = np.random.randint(0, n_vertices, n_edges, dtype=np.uint32)
        targets = np.random.randint(0, n_vertices, n_edges, dtype=np.uint32)
        weights = np.random.random(n_edges).astype(np.float64)

        # Remove self-loops
        mask = sources != targets
        sources = sources[mask]
        targets = targets[mask]
        weights = weights[mask]

        edges_df = pd.DataFrame(
            {"source": sources, "target": targets, "weight": weights}
        )

        print(
            f"  Created random graph with {len(edges_df)} edges, {n_vertices} max vertex ID"
        )

        # Test Louvain performance
        start_time = time.time()
        louvain_result = louvain(edges_df, directed=False)
        louvain_time = time.time() - start_time
        louvain_communities = len(louvain_result["communities"])

        print(f"  ✓ Louvain: {louvain_time:.3f}s, {louvain_communities} communities")

        # Test Leiden performance
        start_time = time.time()
        leiden_result = leiden(edges_df, directed=False)
        leiden_time = time.time() - start_time
        leiden_communities = len(leiden_result["communities"])

        print(f"  ✓ Leiden: {leiden_time:.3f}s, {leiden_communities} communities")

    except Exception as e:
        print(f"    ✗ Performance test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Run all tests."""
    print("=" * 60)
    print("LEIDEN COMMUNITIES PACKAGE TEST SUITE")
    print("=" * 60)

    tests = [
        test_basic_import,
        test_low_level_bindings,
        test_high_level_pandas,
        test_arrow_interface,
        test_edge_cases,
        test_performance,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print(
            "🎉 All tests passed! The leiden_communities package is working correctly."
        )
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
