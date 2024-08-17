import numpy as np
from scipy.optimize import minimize

def compute_centrality(L):
    eigenvalues, eigenvectors = np.linalg.eig(L.T)
    
    # Find the index of the smallest eigenvalue (which should be close to zero)
    min_eigenvalue_index = np.argmin(np.abs(eigenvalues))
    
    # The corresponding eigenvector is used as the centrality vector
    centrality_vector = np.abs(eigenvectors[:, min_eigenvalue_index]).real
    
    # Normalize the centrality vector to sum to 1
    centrality_vector /= centrality_vector.sum()
    
    return centrality_vector

def compute_centrality_second_smallest(L):
    eigenvalues, eigenvectors = np.linalg.eig(L.T)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Use the eigenvector corresponding to the second smallest eigenvalue
    second_smallest_index = 1
    centrality_vector = np.abs(eigenvectors[:, second_smallest_index]).real
    
    # Normalize the centrality vector to sum to 1
    centrality_vector /= centrality_vector.sum()
    
    return centrality_vector

def compute_laplacian(weights, connectivity_matrix):
    weighted_adj_matrix = np.zeros_like(connectivity_matrix, dtype=float)
    weighted_adj_matrix[connectivity_matrix == 1] = weights
    degree_matrix = np.diag(np.sum(weighted_adj_matrix, axis=1))
    laplacian = degree_matrix - weighted_adj_matrix
    return laplacian

def compute_normalized_laplacian(adjacency_matrix):
    # Degree matrix
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    
    # Inverse square root of the degree matrix
    with np.errstate(divide='ignore'):  # Ignore division by zero warnings
        inv_sqrt_degree_matrix = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    
    # Replace infinities with zeros (due to isolated nodes)
    inv_sqrt_degree_matrix[~np.isfinite(inv_sqrt_degree_matrix)] = 0
    
    # Normalized Laplacian
    L_norm = np.eye(adjacency_matrix.shape[0]) - inv_sqrt_degree_matrix @ adjacency_matrix @ inv_sqrt_degree_matrix
    return L_norm

def compute_centrality_normalized(L_norm):
    eigenvalues, eigenvectors = np.linalg.eig(L_norm)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Use the eigenvector corresponding to the second smallest eigenvalue
    second_smallest_index = 1
    centrality_vector = np.abs(eigenvectors[:, second_smallest_index]).real
    
    # Normalize the centrality vector to sum to 1
    centrality_vector /= centrality_vector.sum()
    
    return centrality_vector

def compute_centrality_with_isolated(L):
    eigenvalues, eigenvectors = np.linalg.eig(L.T)
    
    # Find the index of the smallest eigenvalue (which should be close to zero)
    min_eigenvalue_index = np.argmin(np.abs(eigenvalues))
    
    # The corresponding eigenvector is used as the centrality vector
    centrality_vector = np.abs(eigenvectors[:, min_eigenvalue_index]).real
    
    # Normalize the centrality vector to sum to 1
    centrality_vector /= centrality_vector.sum()

    # Detect isolated nodes (rows in L with all zeros) and assign them a centrality of zero
    isolated_nodes = np.all(L == 0, axis=1)
    centrality_vector[isolated_nodes] = 0
    
    # Re-normalize the centrality vector to sum to 1, excluding isolated nodes
    centrality_vector /= centrality_vector.sum()
    
    return centrality_vector

def inverse_centrality(connectivity_matrix, desired_centrality):
    num_edges = np.sum(connectivity_matrix)  # Number of edges (non-zero entries)
    
    if num_edges == 0:
        raise ValueError("The connectivity matrix has no edges.")
    
    initial_weights = np.ones(int(num_edges))  # Start with equal weights for all edges
    
    bounds = [(0, None)] * int(num_edges)  # Weights should be non-negative
    
    def objective(weights, connectivity_matrix, desired_centrality):
        L = compute_laplacian(weights, connectivity_matrix)
        computed_centrality = compute_centrality(L)
        return np.linalg.norm(computed_centrality - desired_centrality)

    result = minimize(
        objective, initial_weights, args=(connectivity_matrix, desired_centrality),
        method='L-BFGS-B', bounds=bounds
    )
    
    optimal_weights = result.x
    weighted_adj_matrix = np.zeros_like(connectivity_matrix, dtype=float)
    
    # Assign weights to the adjacency matrix
    weighted_adj_matrix[connectivity_matrix == 1] = optimal_weights
    
    # Compute the Laplacian and resulting centrality from the optimized adjacency matrix
    L = compute_laplacian(optimal_weights, connectivity_matrix)
    computed_centrality = compute_centrality(L)
    
    return weighted_adj_matrix, computed_centrality

if __name__ == "__main__":
    # Example usage:
    connectivity_matrix = np.array(
        [
            # 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 2
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 7
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 8
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # 10
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],  # 11
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # 12
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 13
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 14
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 15
        ]
    )

    desired_centrality = np.array(
        [0.08, # 1
        0.06, # 2
        0.14, # 3
        0.08, # 4
        0.04, # 5
        0.02, # 6
        0.06, # 7
        0.03, # 8
        0.04, # 9
        0.08, # 10
        0.13, # 11
        0.06, # 12
        0.04, # 13
        0.07, # 14
        0.07] # 15
    )  # Example centrality

    optimized_adj_matrix, resulting_centrality = inverse_centrality(
        connectivity_matrix, desired_centrality
    )
    print("Optimized Weighted Adjacency Matrix:\n", optimized_adj_matrix)
    print("Resulting Centrality:\n", resulting_centrality)