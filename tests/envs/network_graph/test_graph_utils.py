import pytest
import numpy as np

from rl_envs_forge.envs.network_graph.graph_utils import (
    compute_centrality_for_component,
    compute_centrality,
    compute_laplacian,
    get_weighted_adjacency_matrix,
)


class TestGraphUtils:

    def test_compute_centrality_for_component(self):
        L = np.array(
            [
                [1.29864103, -0.64932051, -0.64932052],
                [-1.29864104, 1.29864104, 0.0],
                [-1.29864105, 0.0, 1.29864105],
            ]
        )
        centrality = compute_centrality_for_component(L)
        expected_centrality = np.array([0.5, 0.25, 0.25])
        np.testing.assert_almost_equal(centrality, expected_centrality)

    def test_compute_centrality(self):
        adjacency_matrix = np.array(
            [
                [0, 0.64932051, 0.64932052, 0],
                [
                    1.29864104,
                    0,
                    0,
                    0,
                ],
                [
                    1.29864105,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        )

        L = compute_laplacian(adjacency_matrix)
        centrality = compute_centrality(L, adjacency_matrix)
        expected_centrality = np.array([0.5, 0.25, 0.25, 0.0])
        np.testing.assert_almost_equal(centrality, expected_centrality, decimal=7)

    def test_compute_laplacian(self):
        adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        L = compute_laplacian(adjacency_matrix)
        expected_L = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
        np.testing.assert_array_equal(L, expected_L)

    def test_get_weighted_adjacency_matrix_warning(self):
        connectivity_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        desired_centrality = np.array([0.1, 2, 0.1])
        with pytest.warns(UserWarning):  # Adjust to catch the correct warning type
            weighted_adj_matrix, computed_centrality = get_weighted_adjacency_matrix(
                connectivity_matrix, desired_centrality
            )
        assert weighted_adj_matrix.shape == connectivity_matrix.shape
        assert np.all(computed_centrality >= 0)

    def test_get_weighted_adjacency_matrix_no_edges(self):
        connectivity_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        desired_centrality = np.array([0.3, 0.4, 0.3])
        with pytest.raises(ValueError, match="The connectivity matrix has no edges."):
            get_weighted_adjacency_matrix(connectivity_matrix, desired_centrality)

    def test_random_directed_graph_centrality_varies(self):
        """Test that eigenvector centrality varies in a random directed graph."""
        np.random.seed(42)
        num_nodes = 10
        adj = np.zeros((num_nodes, num_nodes))

        # Generate a directed random graph with varying connections
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and np.random.rand() < 0.3:
                    adj[i, j] = 1

        L = compute_laplacian(adj)
        centrality = compute_centrality(L, adj)

        assert np.isclose(np.sum(centrality), 1.0), "Centrality should be normalized"
        assert np.std(centrality) > 0.01, "Centrality should vary across nodes"
        