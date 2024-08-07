import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def draw_network_graph(adjacency_matrix, centralities, spread_factor=1.0):
    G = nx.DiGraph(adjacency_matrix)
    node_sizes = [5000 * centrality for centrality in centralities]
    
    # Try Kamada-Kawai layout for potentially better spacing
    pos = nx.kamada_kawai_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=20, arrowstyle='->')
    
    for i, j in G.edges():
        if G.has_edge(j, i):
            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], edge_color='gray', arrowsize=20, arrowstyle='<->')

    plt.title("Network Graph with Centrality-Based Node Sizes")
    plt.show()