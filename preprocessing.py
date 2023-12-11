"""
Data pre-processing
Takes images as input and returns TMDs to be used for the MLP

The main function is `preprocess` that takes both image channels and return the TMD
"""

import sys
import numpy as np
from skimage.graph import pixel_graph
import matplotlib.pyplot as plt
from skimage.feature import blob_doh
from skimage import io
import networkx as nx


def compute_shortest_paths(image, positions):
    # Create a graph with input positions as nodes
    graph, nodes = pixel_graph(image)
    graph = nx.from_scipy_sparse_array(graph)

    # Compute the node label for each pixel
    pos_dict = {}
    for index, pos in enumerate(nodes):
        multi_dim_pos = np.unravel_index(pos, image.shape)
        pos_dict[multi_dim_pos] = index

    # Compute the distance between all pairs of nodes in the positions
    distances = np.full((len(positions), len(positions)), np.nan)
    for i, source in enumerate(positions):
        for j, target in list(enumerate(positions))[i + 1:]:
            source_node, target_node = pos_dict[source], pos_dict[target]
            try:
                weight = nx.shortest_path_length(graph, source_node, target_node)
                distances[i][j] = weight
            except nx.exception.NetworkXNoPath:
                distances[i][j] = np.nan

    return distances

def create_graph(image, centers):
    """
    Creates the graph from of the image and the nuclei
    Input:
        - the binary image, with pixels as True or False
        - the list of centers of the blobs detected in the image
    Output:
        - the corresponding tree for this image
    """
    # Compute shortest paths between input positions
    distances = compute_shortest_paths(image, centers)

    # Create a graph with 1..n nodes
    graph = nx.Graph()
    graph.add_nodes_from(range(len(centers)))

    # Add weighted edges to the graph
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            weight = distances[i][j]
            if not np.isnan(weight):  # Ignore infinite weights
                graph.add_edge(i, j, weight=weight)

    # Find the minimum spanning tree
    mst = nx.minimum_spanning_tree(graph)

    return mst


def plot_graph_on_image(image, centers, graph):
    """
    Plots the graph on top of the original image
    using the centers to get the positions of the nodes in the image
    Input:
        - the binary image of the neurites
        - the list of positions for each node, in the order of the node number
        - the graph to display
    """
    # Swap x and y for correct positions with matplotlib
    pos = {
        i: (y, x) for i, (x, y) in enumerate(centers)
    }

    plt.imshow(image, cmap="gray")
    nx.draw(graph, pos, node_color="g", node_size=100, edge_color="r")
    plt.show()


def preprocess(nuclei_img, dendrites_img, graphical=False):
    """
    Pre-processes the nuclei and dendrites image (in their binary array format)
    and returns the corresponding features
    """
    # Find blobs
    print("Finding the nuclei...")
    blobs = blob_doh(nuclei_img, min_sigma=30, max_sigma=80)

    neuron_centers = [(int(x), int(y)) for (x, y, area) in blobs]

    merged_img = (nuclei_img != 0) | (dendrites_img != 0)

    print("Creating the graph...")
    graph = create_graph(merged_img, neuron_centers)

    if graphical:
        # Plot the resulting graph
        plot_graph_on_image(dendrites_img, neuron_centers, graph)

    # TODO: extract topological features
    print("Extracting the topological features...")

if len(sys.argv) != 3:
    print("Usage: python preprocessing.py [nuclei_file_path] [dendrites_file_path]")
    sys.exit(1)

# Extract the two arguments
nuclei_file = io.imread(sys.argv[1])
dendrites_file = io.imread(sys.argv[2])

preprocess(nuclei_file, dendrites_file, graphical=True)
