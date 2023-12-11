"""
Data pre-processing
Converts binary images of nuclei and dendrites into a Minimum Spanning Tree (MST) graph representation.

The main function is `preprocess`, which takes both binary image channels and returns the corresponding topological features.
"""

import sys
import numpy as np
from skimage.graph import pixel_graph
import matplotlib.pyplot as plt
from skimage.feature import blob_doh
from skimage import io
import networkx as nx


def compute_shortest_paths(image, positions):
    """
    Computes the shortest paths between given positions on a pixel graph.
    Input:
        - image: Binary image representing the graph structure.
        - positions: List of positions (nodes) on which to compute shortest paths.
    Output:
        - distances: 2D array containing distances between each pair of positions.
    """
    # Create a graph with input positions as nodes
    graph, nodes = pixel_graph(image)
    graph = nx.from_scipy_sparse_array(graph)

    # Compute the node label for each pixel
    pos_dict = {
        np.unravel_index(pos, image.shape): index for index, pos in enumerate(nodes)
    }

    # Compute the distance between all pairs of nodes in the positions
    distances = np.full((len(positions), len(positions)), np.nan)
    for i, source in enumerate(positions):
        for j, target in enumerate(positions[i + 1 :]):
            source_node, target_node = pos_dict[source], pos_dict[target]
            try:
                weight = nx.shortest_path_length(graph, source_node, target_node)
                distances[i][j + i + 1] = distances[j + i + 1][i] = weight
            except nx.exception.NetworkXNoPath:
                distances[i][j + i + 1] = distances[j + i + 1][i] = np.nan

    return distances


def create_graph(image, centers):
    """
    Creates a Minimum Spanning Tree (MST) graph from the binary image and the nuclei positions.
    Input:
        - image: Binary image with pixels as True or False.
        - centers: List of centers of the blobs detected in the image.
    Output:
        - mst: Minimum Spanning Tree graph.
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
    Plots the Minimum Spanning Tree (MST) graph on top of the original image.
    Input:
        - image: Binary image of the neurites.
        - centers: List of positions for each node, in the order of the node number.
        - graph: Minimum Spanning Tree (MST) to display.
    """
    # Swap x and y for correct positions with matplotlib
    pos = {i: (y, x) for i, (x, y) in enumerate(centers)}

    plt.imshow(image, cmap="gray")
    nx.draw(graph, pos, node_color="g", node_size=100, edge_color="r")
    plt.show()

def insert_circles(image, positions, radius):
    """
    Inserts circles of "True" in a 2D numpy array around given positions (in place).
    Input:
        - image: 2D numpy array of True/False.
        - positions: List of (row, column) positions where circles will be inserted.
        - radius: Radius of the circles to be inserted.
    """
    rows, cols = image.shape
    y, x = np.ogrid[:rows, :cols]

    for pos in positions:
        circle_mask = (x - pos[1]) ** 2 + (y - pos[0]) ** 2 <= radius ** 2
        image[circle_mask] = True

def preprocess(nuclei_img, dendrites_img, graphical=False):
    """
    Pre-processes the nuclei and dendrites images (in their binary array format)
    and returns the corresponding topological features.
    """
    # Find blobs
    print("Finding the nuclei...")
    blobs = blob_doh(nuclei_img, min_sigma=30, max_sigma=80)
    neuron_centers = [(int(x), int(y)) for (x, y, _) in blobs]

    print("Refining the image...")
    merged_img = (nuclei_img != 0) | (dendrites_img != 0)
    insert_circles(merged_img, neuron_centers, 10)

    print("Creating the graph...")
    graph = create_graph(merged_img, neuron_centers)

    if graphical:
        # Plot the resulting graph
        plot_graph_on_image(dendrites_img, neuron_centers, graph)

    # TODO: Extract topological features
    print("Extracting the topological features...")


if len(sys.argv) != 3:
    print("Usage: python preprocessing.py [nuclei_file_path] [dendrites_file_path]")
    sys.exit(1)

# Extract the two arguments
nuclei_file = io.imread(sys.argv[1])
dendrites_file = io.imread(sys.argv[2])

preprocess(nuclei_file, dendrites_file, graphical=True)
