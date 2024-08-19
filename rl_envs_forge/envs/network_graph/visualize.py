import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def draw_network_graph(adjacency_matrix, node_sizes):
    G = nx.DiGraph(adjacency_matrix)
    node_sizes = [5000 * centrality for centrality in node_sizes]

    # Use Spring layout with adjusted k parameter to prevent node overlap
    pos = nx.spring_layout(G, k=2/(len(G)**0.5), iterations=50)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=20, arrowstyle='->')
    
    for i, j in G.edges():
        if G.has_edge(j, i):
            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], edge_color='gray', arrowsize=20, arrowstyle='<->')

    plt.title("Network Graph")
    plt.show()