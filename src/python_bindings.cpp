#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <vector>
#include <memory>
#include <cstdint>
#include <chrono>

// Include the main header files from the project
#include "main.hxx"
#include "properties.hxx"
#include "update.hxx"
#include "_openmp.hxx"

namespace py = pybind11;
using namespace std;

// Type aliases for consistency with the main codebase
using K = uint32_t;
using V = float;
using W = double;

/**
 * Convert pandas DataFrame-style edge list to DiGraph
 * Expected format: DataFrame with columns 'source', 'target', 'weight' (optional)
 * Input arrays are expected to be numpy arrays from pandas
 */
template<typename G>
G create_graph_from_arrays(
    py::array_t<uint32_t> sources,
    py::array_t<uint32_t> targets,
    py::array_t<double> weights = py::array_t<double>(),
    bool directed = true
) {
    auto src_buf = sources.request();
    auto tgt_buf = targets.request();

    if (src_buf.size != tgt_buf.size) {
        throw std::invalid_argument("Source and target arrays must have the same length");
    }

    auto src_ptr = static_cast<uint32_t*>(src_buf.ptr);
    auto tgt_ptr = static_cast<uint32_t*>(tgt_buf.ptr);

    bool has_weights = weights.size() > 0;
    double* weight_ptr = nullptr;

    if (has_weights) {
        auto weight_buf = weights.request();
        if (weight_buf.size != src_buf.size) {
            throw std::invalid_argument("Weight array must have the same length as source/target arrays");
        }
        weight_ptr = static_cast<double*>(weight_buf.ptr);
    }

    // Find the maximum vertex ID to determine graph span
    uint32_t max_vertex = 0;
    #ifdef OPENMP
    #pragma omp parallel for schedule(dynamic, 2048) reduction(max:max_vertex)
    #endif
    for (py::ssize_t i = 0; i < src_buf.size; ++i) {
        max_vertex = std::max(max_vertex, std::max(src_ptr[i], tgt_ptr[i]));
    }

    G graph;
    using K = typename G::key_type;
    using V_type = typename G::vertex_value_type;
    using E = typename G::edge_value_type;

    // Use the same pattern as readMtxIfOmpW: add vertices first, then edges in parallel
    auto fv = [](auto u, auto d) { return true; };
    addVerticesIfU(graph, K(0), K(max_vertex + 1), V_type(), fv);

#ifdef OPENMP
    // Process edges in batches similar to readMtxDoOmp
    const py::ssize_t BATCH_SIZE = 131072;
    py::ssize_t total_edges = src_buf.size;

    for (py::ssize_t batch_start = 0; batch_start < total_edges; batch_start += BATCH_SIZE) {
        py::ssize_t batch_end = std::min(batch_start + BATCH_SIZE, total_edges);
        py::ssize_t batch_size = batch_end - batch_start;

        // Process this batch in parallel
        #pragma omp parallel
        {
            // Each thread processes edges from this batch
            for (py::ssize_t i = batch_start; i < batch_end; ++i) {
                uint32_t u = src_ptr[i];
                uint32_t v = tgt_ptr[i];
                double w = has_weights ? weight_ptr[i] : 1.0;

                // Use addEdgeOmpU for thread-safe edge addition
                addEdgeOmpU(graph, K(u), K(v), E(w));

                // Add reverse edge for undirected graphs
                if (!directed && u != v) {
                    addEdgeOmpU(graph, K(v), K(u), E(w));
                }
            }
        }
    }

    // Use updateOmpU for thread-safe graph update
    updateOmpU(graph);
#else
    // Fallback to sequential processing when OpenMP is not available
    for (py::ssize_t i = 0; i < src_buf.size; ++i) {
        uint32_t u = src_ptr[i];
        uint32_t v = tgt_ptr[i];
        double w = has_weights ? weight_ptr[i] : 1.0;

        graph.addEdge(K(u), K(v), E(w));

        // Add reverse edge for undirected graphs
        if (!directed && u != v) {
            graph.addEdge(K(v), K(u), E(w));
        }
    }

    graph.update();
#endif
    return graph;
}

/**
 * Convert LouvainResult to Python dictionary with Arrow-compatible arrays
 */
py::dict louvain_result_to_dict(const LouvainResult<K, W>& result) {
    py::dict output;

    // Convert membership vector to numpy array
    size_t n_vertices = result.membership.size();
    auto membership_array = py::array_t<uint32_t>(n_vertices);
    auto membership_buf = membership_array.request();
    auto membership_ptr = static_cast<uint32_t*>(membership_buf.ptr);

    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < n_vertices; ++i) {
        membership_ptr[i] = result.membership[i];
    }

    // Convert vertex weights to numpy array
    auto vertex_weights_array = py::array_t<double>(n_vertices);
    auto vertex_weights_buf = vertex_weights_array.request();
    auto vertex_weights_ptr = static_cast<double*>(vertex_weights_buf.ptr);

    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < n_vertices; ++i) {
        vertex_weights_ptr[i] = result.vertexWeight[i];
    }

    // Convert community weights to numpy array
    size_t n_communities = result.communityWeight.size();
    auto community_weights_array = py::array_t<double>(n_communities);
    auto community_weights_buf = community_weights_array.request();
    auto community_weights_ptr = static_cast<double*>(community_weights_buf.ptr);

    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < n_communities; ++i) {
        community_weights_ptr[i] = result.communityWeight[i];
    }

    output["membership"] = membership_array;
    output["vertex_weights"] = vertex_weights_array;
    output["community_weights"] = community_weights_array;
    output["iterations"] = result.iterations;
    output["passes"] = result.passes;
    output["time"] = result.time;
    output["marking_time"] = result.markingTime;
    output["initialization_time"] = result.initializationTime;
    output["first_pass_time"] = result.firstPassTime;
    output["local_move_time"] = result.localMoveTime;
    output["aggregation_time"] = result.aggregationTime;
    output["affected_vertices"] = result.affectedVertices;

    return output;
}

/**
 * Convert LeidenResult to Python dictionary with Arrow-compatible arrays
 */
py::dict leiden_result_to_dict(const LeidenResult<K, W>& result) {
    py::dict output;

    // Convert membership vector to numpy array
    size_t n_vertices = result.membership.size();
    auto membership_array = py::array_t<uint32_t>(n_vertices);
    auto membership_buf = membership_array.request();
    auto membership_ptr = static_cast<uint32_t*>(membership_buf.ptr);

    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < n_vertices; ++i) {
        membership_ptr[i] = result.membership[i];
    }

    // Convert vertex weights to numpy array
    auto vertex_weights_array = py::array_t<double>(n_vertices);
    auto vertex_weights_buf = vertex_weights_array.request();
    auto vertex_weights_ptr = static_cast<double*>(vertex_weights_buf.ptr);

    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < n_vertices; ++i) {
        vertex_weights_ptr[i] = result.vertexWeight[i];
    }

    // Convert community weights to numpy array
    size_t n_communities = result.communityWeight.size();
    auto community_weights_array = py::array_t<double>(n_communities);
    auto community_weights_buf = community_weights_array.request();
    auto community_weights_ptr = static_cast<double*>(community_weights_buf.ptr);

    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < n_communities; ++i) {
        community_weights_ptr[i] = result.communityWeight[i];
    }

    output["membership"] = membership_array;
    output["vertex_weights"] = vertex_weights_array;
    output["community_weights"] = community_weights_array;
    output["iterations"] = result.iterations;
    output["passes"] = result.passes;
    output["time"] = result.time;
    output["marking_time"] = result.markingTime;
    output["initialization_time"] = result.initializationTime;
    output["first_pass_time"] = result.firstPassTime;
    output["local_move_time"] = result.localMoveTime;
    output["refinement_time"] = result.refinementTime;
    output["aggregation_time"] = result.aggregationTime;
    output["affected_vertices"] = result.affectedVertices;

    return output;
}

/**
 * Python wrapper for Louvain algorithm
 */
py::dict py_louvain(
    py::array_t<uint32_t> sources,
    py::array_t<uint32_t> targets,
    py::array_t<double> weights = py::array_t<double>(),
    bool directed = true,
    int repeat = 1,
    double resolution = 1.0,
    double tolerance = 1e-2,
    double aggregation_tolerance = 0.8,
    double tolerance_drop = 10.0,
    int max_iterations = 20,
    int max_passes = 10
) {
    // Create graph from input arrays
    using GraphType = DiGraph<K, None, V>;
    auto graph = create_graph_from_arrays<GraphType>(sources, targets, weights, directed);

    // Set up Louvain options
    LouvainOptions options(
        repeat,
        resolution,
        tolerance,
        aggregation_tolerance,
        tolerance_drop,
        max_iterations,
        max_passes
    );

    // Run Louvain algorithm
    auto result = louvainStaticOmp(graph, options);

    // Convert result to Python dictionary
    return louvain_result_to_dict(result);
}

/**
 * Python wrapper for Leiden algorithm
 */
py::dict py_leiden(
    py::array_t<uint32_t> sources,
    py::array_t<uint32_t> targets,
    py::array_t<double> weights = py::array_t<double>(),
    bool directed = true,
    int repeat = 1,
    double resolution = 1.0,
    double tolerance = 1e-2,
    double aggregation_tolerance = 0.8,
    double tolerance_drop = 10.0,
    int max_iterations = 20,
    int max_passes = 10
) {
    // Create graph from input arrays
    using GraphType = DiGraph<K, None, V>;
    auto graph = create_graph_from_arrays<GraphType>(sources, targets, weights, directed);

    // Set up Leiden options
    LeidenOptions options(
        repeat,
        resolution,
        tolerance,
        aggregation_tolerance,
        tolerance_drop,
        max_iterations,
        max_passes
    );

    // Run Leiden algorithm
    auto result = leidenStaticOmp(graph, options);

    // Convert result to Python dictionary
    return leiden_result_to_dict(result);
}

/**
 * Python wrapper for modularity calculation from arrays using modularityByOmp
 */
double py_modularity(
    py::array_t<uint32_t> sources,
    py::array_t<uint32_t> targets,
    py::array_t<uint32_t> membership,
    py::array_t<double> weights = py::array_t<double>(),
    bool directed = true,
    double resolution = 1.0
) {
    // Create graph from input arrays
    using GraphType = DiGraph<K, None, V>;
    auto graph = create_graph_from_arrays<GraphType>(sources, targets, weights, directed);

    // Get membership data
    const uint32_t* membership_ptr = membership.data();
    size_t num_vertices = membership.size();

    // Create community assignment function
    auto fc = [membership_ptr](uint32_t u) -> uint32_t {
        return membership_ptr[u];
    };

    // Calculate total edge weight (M = total weight / 2 for undirected graphs)
    double M = edgeWeightOmp(graph);
    if (!directed) {
        M /= 2.0;
    }

    // Calculate modularity using modularityByOmp
    return modularityByOmp(graph, fc, M, resolution);
}

/**
 * OpenMP utility functions
 */
#ifdef OPENMP
int get_max_threads() {
    return omp_get_max_threads();
}

int get_num_threads() {
    // omp_get_num_threads() returns 1 outside parallel regions
    // So we use omp_get_max_threads() which gives the number of threads
    // that would be used in the next parallel region
    return omp_get_max_threads();
}

void set_num_threads(int num_threads) {
    omp_set_num_threads(num_threads);
}
#else
int get_max_threads() { return 1; }
int get_num_threads() { return 1; }
void set_num_threads(int num_threads) { /* No-op */ }
#endif

PYBIND11_MODULE(_leiden_cpp, m) {
    m.doc() = "High-performance Leiden and Louvain community detection algorithms";

    m.def("louvain", &py_louvain,
          "Run Louvain community detection algorithm",
          py::arg("sources"),
          py::arg("targets"),
          py::arg("weights") = py::array_t<double>(),
          py::arg("directed") = true,
          py::arg("repeat") = 1,
          py::arg("resolution") = 1.0,
          py::arg("tolerance") = 1e-2,
          py::arg("aggregation_tolerance") = 0.8,
          py::arg("tolerance_drop") = 10.0,
          py::arg("max_iterations") = 20,
          py::arg("max_passes") = 10);

    m.def("leiden", &py_leiden,
          "Run Leiden community detection algorithm",
          py::arg("sources"),
          py::arg("targets"),
          py::arg("weights") = py::array_t<double>(),
          py::arg("directed") = true,
          py::arg("repeat") = 1,
          py::arg("resolution") = 1.0,
          py::arg("tolerance") = 1e-2,
          py::arg("aggregation_tolerance") = 0.8,
          py::arg("tolerance_drop") = 10.0,
          py::arg("max_iterations") = 20,
          py::arg("max_passes") = 10);

    m.def("modularity", &py_modularity,
          "Calculate modularity from edge arrays and community membership",
          py::arg("sources"),
          py::arg("targets"),
          py::arg("membership"),
          py::arg("weights") = py::array_t<double>(),
          py::arg("directed") = true,
          py::arg("resolution") = 1.0);

    // Add OpenMP thread control functions
    m.def("get_max_threads", &get_max_threads,
          "Get the maximum number of OpenMP threads available");

    m.def("get_num_threads", &get_num_threads,
          "Get the current number of OpenMP threads");

    m.def("set_num_threads", &set_num_threads,
          "Set the number of OpenMP threads to use",
          py::arg("num_threads"));
}
