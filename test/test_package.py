#!/usr/bin/env python3
"""
Simple tests for the leiden_communities package.
"""

import os
import sys
import unittest

import pandas as pd

# Add package to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import leiden_communities as lc

    PACKAGE_AVAILABLE = True
except ImportError as e:
    print(f"Package not available: {e}")
    PACKAGE_AVAILABLE = False


class TestLeidenCommunities(unittest.TestCase):
    """Test cases for leiden_communities package."""

    def setUp(self):
        """Set up test data."""
        if not PACKAGE_AVAILABLE:
            self.skipTest("Package not built")

        # Create a simple test graph
        self.test_graph = pd.DataFrame(
            {
                "source": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                "target": [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
                "weight": [1.0] * 10,
            }
        )

    def test_louvain_basic(self):
        """Test basic Louvain functionality."""
        result = lc.louvain(self.test_graph, directed=False)

        self.assertIn("vertices", result)
        self.assertIn("communities", result)
        self.assertIn("stats", result)

        vertices_df = result["vertices"]
        self.assertEqual(len(vertices_df), 6)  # 6 unique vertices
        self.assertTrue(
            all(
                col in vertices_df.columns
                for col in ["vertex", "community", "vertex_weight"]
            )
        )

    def test_leiden_basic(self):
        """Test basic Leiden functionality."""
        result = lc.leiden(self.test_graph, directed=False)

        self.assertIn("vertices", result)
        self.assertIn("communities", result)
        self.assertIn("stats", result)

        vertices_df = result["vertices"]
        self.assertEqual(len(vertices_df), 6)  # 6 unique vertices
        self.assertTrue(
            all(
                col in vertices_df.columns
                for col in ["vertex", "community", "vertex_weight"]
            )
        )

    def test_from_edge_list(self):
        """Test edge list creation."""
        edges = [(0, 1), (1, 2), (2, 0)]
        df = lc.from_edge_list(edges)

        self.assertEqual(len(df), 3)
        self.assertTrue(
            all(col in df.columns for col in ["source", "target", "weight"])
        )
        self.assertTrue(all(df["weight"] == 1.0))

    def test_parameter_validation(self):
        """Test input validation."""
        # Test missing columns
        bad_df = pd.DataFrame({"src": [0], "dst": [1]})
        with self.assertRaises(ValueError):
            lc.louvain(bad_df)

        # Test empty DataFrame
        empty_df = pd.DataFrame(columns=["source", "target"])
        with self.assertRaises(ValueError):
            lc.leiden(empty_df)

    def test_directed_vs_undirected(self):
        """Test directed vs undirected graphs."""
        result_directed = lc.louvain(self.test_graph, directed=True)
        result_undirected = lc.louvain(self.test_graph, directed=False)

        # Both should work without errors
        self.assertIn("vertices", result_directed)
        self.assertIn("vertices", result_undirected)

    def test_arrow_conversion(self):
        """Test Arrow Table conversion."""
        result = lc.leiden(self.test_graph, directed=False, return_arrow=False)
        arrow_result = lc.to_arrow_table(result)

        # Check that conversion worked
        import pyarrow as pa

        self.assertIsInstance(arrow_result["vertices"], pa.Table)
        self.assertIsInstance(arrow_result["communities"], pa.Table)

        # Convert back to pandas
        pandas_result = lc.to_pandas(arrow_result)
        self.assertIsInstance(pandas_result["vertices"], pd.DataFrame)


def simple_functional_test():
    """Simple functional test that doesn't require unittest."""
    print("Running simple functional test...")

    if not PACKAGE_AVAILABLE:
        print("SKIP: Package not available")
        return False

    try:
        # Create test graph
        df = pd.DataFrame(
            {
                "source": [0, 1, 2, 3],
                "target": [1, 2, 0, 1],
                "weight": [1.0, 1.0, 0.5, 0.1],
            }
        )

        # Test Louvain
        louvain_result = lc.louvain(df, directed=False)
        print(f"Louvain: Found {len(louvain_result['communities'])} communities")

        # Test Leiden
        leiden_result = lc.leiden(df, directed=False)
        print(f"Leiden: Found {len(leiden_result['communities'])} communities")

        # Test edge list
        edges = [(0, 1), (1, 2)]
        edge_df = lc.from_edge_list(edges)
        print(f"Edge list: Created graph with {len(edge_df)} edges")

        print("SUCCESS: All basic functions working")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


if __name__ == "__main__":
    # Run simple test first
    if simple_functional_test():
        print("\n" + "=" * 50)
        print("Running detailed tests...")
        unittest.main(verbosity=2)
    else:
        print("Simple test failed, skipping detailed tests")
        sys.exit(1)
