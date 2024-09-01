import numpy as np
import warnings
from scipy.optimize import minimize


def normalize_adjacency_matrix(A):
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_normalized = A / row_sums
    return A_normalized

# Compute Laplacian
def compute_laplacian(adjacency_matrix):
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian = degree_matrix - adjacency_matrix
    return laplacian

# Compute centrality from Laplacian
def compute_eigenvector_centrality(L):
    eigenvalues, eigenvectors = np.linalg.eig(L.T)  # Transpose for left eigenvector
    min_eigenvalue_index = np.argmin(np.abs(eigenvalues))
    eigv = np.real(eigenvectors[:, min_eigenvalue_index])
    eigv_normalized = np.abs(eigv) / np.sum(np.abs(eigv))
    return eigv_normalized


def process_multiple_components(components):
    """
    Processes multiple components and computes the centrality for each.

    Parameters:
    components (list of numpy.ndarray): List of adjacency matrices for each component.

    Returns:
    list of numpy.ndarray: Normalized eigenvector centralities for each component.
    """
    centralities = []
    for component in components:
        L_component = compute_laplacian(component)
        centrality = compute_eigenvector_centrality(L_component)
        centralities.append(centrality)
    return centralities


def compute_centrality_for_component(L):
    """
    Computes centrality for a single connected component.

    Parameters:
    L (numpy.ndarray): Laplacian matrix of the connected component.

    Returns:
    numpy.ndarray: The computed centrality vector for the component.
    """
    eigenvalues, eigenvectors = np.linalg.eig(L.T)

    # Find the index of the smallest eigenvalue (which should be close to zero)
    min_eigenvalue_index = np.argmin(np.abs(eigenvalues))

    # The corresponding eigenvector is used as the centrality vector
    centrality_vector = np.abs(eigenvectors[:, min_eigenvalue_index]).real

    # Normalize the centrality vector to sum to 1
    if np.sum(centrality_vector) > 0:
        centrality_vector /= np.sum(centrality_vector)

    return centrality_vector


def compute_centrality(L, adjacency_matrix):
    """
    Computes centrality for both connected and disconnected graphs by
    computing centralities for each connected component.

    Parameters:
    L (numpy.ndarray): Laplacian matrix of the graph.
    adjacency_matrix (numpy.ndarray): Adjacency matrix of the graph.

    Returns:
    numpy.ndarray: The computed centrality vector for the entire graph.
    """
    num_nodes = L.shape[0]
    centrality = np.zeros(num_nodes)
    visited = np.zeros(num_nodes, dtype=bool)

    # Identify connected components
    for i in range(num_nodes):
        if not visited[i]:
            # Find the nodes in the same connected component as node i
            component = np.where(
                np.linalg.matrix_power(adjacency_matrix + np.eye(num_nodes), num_nodes)[
                    i
                ]
                > 0
            )[0]
            component_L = L[np.ix_(component, component)]
            component_centrality = compute_centrality_for_component(component_L)

            # Check if the entire component is isolated (no incoming or outgoing edges)
            has_incoming_edges = np.any(adjacency_matrix[:, component], axis=0)
            is_isolated_component = not np.any(has_incoming_edges)

            if not is_isolated_component:
                # Assign centrality to non-isolated nodes
                centrality[component] = component_centrality
            visited[component] = True

    # Normalize the centrality vector for non-isolated components
    connected_centrality_sum = np.sum(centrality)
    if connected_centrality_sum > 0:
        centrality /= connected_centrality_sum

    return centrality


def get_weighted_adjacency_matrix(
    connectivity_matrix, desired_centrality, tolerance=0.0001
):
    num_edges = np.sum(connectivity_matrix)  # Number of edges (non-zero entries)

    if num_edges == 0:
        raise ValueError("The connectivity matrix has no edges.")

    initial_weights = np.ones(int(num_edges))  # Start with equal weights for all edges

    bounds = [(0, None)] * int(num_edges)  # Weights should be non-negative

    def objective(weights, connectivity_matrix, desired_centrality):
        weighted_adjacency_matrix = np.zeros_like(connectivity_matrix, dtype=float)
        weighted_adjacency_matrix[connectivity_matrix == 1] = weights
        L = compute_laplacian(weighted_adjacency_matrix)
        computed_centrality = compute_centrality(L, weighted_adjacency_matrix)
        return np.linalg.norm(computed_centrality - desired_centrality)

    result = minimize(
        objective,
        initial_weights,
        args=(connectivity_matrix, desired_centrality),
        method="L-BFGS-B",
        bounds=bounds,
    )

    optimal_weights = result.x
    weighted_adj_matrix = np.zeros_like(connectivity_matrix, dtype=float)

    # Assign weights to the adjacency matrix
    weighted_adj_matrix[connectivity_matrix == 1] = optimal_weights

    # Compute the Laplacian and resulting centrality from the optimized adjacency matrix
    L = compute_laplacian(weighted_adj_matrix)
    computed_centrality = compute_centrality(L, weighted_adj_matrix)

    # Check if the computed centrality is close to the desired centrality
    if not np.allclose(computed_centrality, desired_centrality, atol=tolerance):
        warnings.warn(
            f"Resulting centrality is not within the tolerance of {tolerance}. "
            f"Difference: {np.abs(computed_centrality - desired_centrality)}",
            UserWarning,
        )

    return weighted_adj_matrix, computed_centrality
