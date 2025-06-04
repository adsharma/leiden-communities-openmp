# Python Bindings for Leiden Communities

High-performance Python bindings for the GVE-Leiden and Louvain community detection algorithms with OpenMP parallelization. Built for large-scale network analysis with pandas DataFrame and Apache Arrow support.

This package provides Python access to **GVE-Leiden**, an optimized parallel implementation that surpasses other implementations by significant margins while maintaining algorithm correctness.

## Features

- **High Performance**: C++ implementation with OpenMP parallelization
- **Simple Interface**: `result = louvain(G)` and `result = leiden(G)`
- **Pandas Integration**: Native support for pandas DataFrames
- **Apache Arrow Support**: Zero-copy operations with Arrow Tables
- **Memory Efficient**: Avoids inefficient vector copying and memcopy operations
- **Full Algorithm Control**: Access to all Leiden and Louvain parameters

## Installation

### Prerequisites

- Python 3.8+
- C++ compiler with OpenMP support
- CMake (for building)

### From Source

```bash
git clone <repository-url>
cd leiden-communities-openmp
uv sync
uv pip install -e .
source .venv/bin/activate
python3 demo.py
```

### Dependencies

The package automatically installs:
- `pandas` - DataFrame support
- `pyarrow` - Arrow Table support
- `numpy` - Numerical arrays

Build dependencies (handled automatically):
- `pybind11` - C++ bindings
- `setuptools` - Build system
- `wheel` - Package building

## Quick Start

```python
import pandas as pd
from leiden_communities import louvain, leiden

# Create a graph from edge list
edges = pd.DataFrame({
    'source': [0, 1, 1, 2, 2, 3],
    'target': [1, 2, 0, 3, 1, 2],
    'weight': [1.0, 1.0, 0.5, 1.0, 0.5, 1.0]
})

# Run Louvain algorithm
result = louvain(edges, directed=False, resolution=1.0)
print(f"Found {len(result['communities'])} communities")
print(result['vertices'])  # Vertex community assignments
print(result['communities'])  # Community information

# Run Leiden algorithm
result = leiden(edges, directed=False, resolution=1.0)
print(result['vertices'])
print(result['stats'])  # Algorithm statistics
```

## API Reference

### Core Functions

#### `louvain(graph, **kwargs)`

Run the Louvain community detection algorithm.

**Parameters:**
- `graph`: Input graph as pandas DataFrame, Arrow Table, or dict-like object
- `directed`: bool, default True - Whether graph is directed
- `resolution`: float, default 1.0 - Resolution parameter
- `seed`: int, optional - Random seed for reproducibility
- `max_iterations`: int, default 20 - Maximum iterations per pass
- `max_passes`: int, default 10 - Maximum number of passes
- `return_arrow`: bool, default False - Return Arrow Tables instead of DataFrames

**Returns:**
Dictionary with:
- `'vertices'`: DataFrame/Table with vertex community assignments
- `'communities'`: DataFrame/Table with community information
- `'stats'`: Dictionary with algorithm statistics and timing

#### `leiden(graph, **kwargs)`

Run the Leiden community detection algorithm.

**Parameters:** Same as `louvain()` plus:
- `beta`: float, default 0.01 - Leiden beta parameter

**Returns:** Same structure as `louvain()`

### Utility Functions

#### `from_edge_list(edges, weights=None, directed=True)`

Create a graph DataFrame from an edge list.

**Parameters:**
- `edges`: List of (source, target) tuples or 2D array
- `weights`: Optional list/array of edge weights
- `directed`: bool, default True - Whether graph is directed

**Returns:** pandas DataFrame with 'source', 'target', 'weight' columns

#### `to_arrow_table(result)`

Convert algorithm results to Apache Arrow format.

#### `to_pandas(arrow_table_or_result)`

Convert Arrow Tables back to pandas DataFrames.

## Input Formats

The algorithms accept graphs in multiple formats:

### 1. Pandas DataFrame
```python
edges = pd.DataFrame({
    'source': [0, 1, 2],
    'target': [1, 2, 0],
    'weight': [1.0, 1.0, 0.5]
})
result = louvain(edges)
```

### 2. Apache Arrow Table
```python
import pyarrow as pa
edges_arrow = pa.Table.from_pandas(edges)
result = louvain(edges_arrow, return_arrow=True)
```

### 3. Edge List
```python
from leiden_communities import from_edge_list

edge_list = [(0, 1), (1, 2), (2, 0)]
edges = from_edge_list(edge_list, weights=[1.0, 1.0, 0.5])
result = louvain(edges)
```

### 4. Dict-like Objects
Any object with 'source', 'target', and optionally 'weight' attributes:
```python
class GraphData:
    def __init__(self):
        self.source = [0, 1, 2]
        self.target = [1, 2, 0]
        self.weight = [1.0, 1.0, 0.5]

result = louvain(GraphData())
```

## Output Format

Both algorithms return a dictionary with:

### Vertices DataFrame
```
   vertex  community  vertex_weight
0       0          0            2.3
1       1          0            3.0
2       2          1            1.5
```

### Communities DataFrame
```
   community  community_weight
0          0               5.3
1          1               1.5
```

### Statistics Dictionary
```python
{
    'iterations': 3,
    'passes': 2,
    'total_time': 11.9,
    'affected_vertices': 0,
    'marking_time': 0.134,
    'initialization_time': 0.552,
    'first_pass_time': 9.627,
    'local_move_time': 1.357,
    'aggregation_time': 4.453,
    'refinement_time': 4.232  # Leiden only
}
```

## Performance

The bindings are optimized for performance with GVE-Leiden implementation:

- **OpenMP Parallelization**: Automatic multi-threading
- **Zero-Copy Operations**: Direct numpy array access
- **Apache Arrow Integration**: Efficient columnar data transfer
- **Memory Efficient**: Minimal data copying

### Benchmarks

Example performance on random graphs:
- 500 vertices, ~2000 edges:
  - Louvain: ~0.05s
  - Leiden: ~0.07s

The underlying GVE-Leiden implementation achieves significant speedups over other implementations.

## Advanced Usage

### Algorithm Parameters

```python
# Fine-tune algorithm behavior
result = leiden(
    graph=edges,
    directed=False,
    resolution=0.8,          # Lower = larger communities
    beta=0.01,              # Leiden refinement parameter
    seed=42,                # Reproducible results
    max_iterations=50,      # More thorough optimization
    max_passes=20
)
```

### Apache Arrow Integration

```python
import pyarrow as pa
from leiden_communities import to_arrow_table, to_pandas

# Convert input to Arrow
edges_arrow = pa.Table.from_pandas(edges)

# Run algorithm with Arrow output
result = louvain(edges_arrow, return_arrow=True)

# Access Arrow tables directly
vertices_table = result['vertices']  # pyarrow.Table
communities_table = result['communities']  # pyarrow.Table

# Convert back to pandas when needed
vertices_df = to_pandas(vertices_table)
```

### Error Handling

```python
try:
    result = louvain(edges)
except ImportError:
    print("C++ extension not available")
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Running the Demo

```bash
python demo.py
```

This will demonstrate all functionality including:
- Basic usage with pandas DataFrames
- Apache Arrow integration
- Utility functions
- Performance testing on larger graphs

## Algorithm Details

### Louvain Algorithm
- **Purpose**: Fast community detection for large networks
- **Complexity**: O(n log n) for sparse networks
- **Method**: Iterative local optimization and aggregation

### Leiden Algorithm
- **Purpose**: Improved quality over Louvain with guaranteed connectivity
- **Complexity**: Similar to Louvain with refinement overhead
- **Method**: Louvain with additional refinement phase

### Parameters
- **Resolution**: Controls community size (higher = smaller communities)
- **Beta**: Leiden-specific refinement strength
- **Iterations**: Controls optimization thoroughness vs speed

## References

- Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing well-connected communities. Sci Rep 9, 5233 (2019).
- Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. Journal of statistical mechanics: theory and experiment, 2008(10), P10008.

For more details about the underlying GVE-Leiden implementation, see the main README.md.
