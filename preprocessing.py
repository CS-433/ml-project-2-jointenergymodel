"""
Data pre-processing
Converts binary images of nuclei and dendrites into a Minimum Spanning Tree (MST) graph representation.

The main function is `preprocess`, which takes both binary image channels and returns the corresponding topological features.
"""

import sys
import os
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import pixel_graph
from scipy.ndimage import binary_dilation
from skimage.feature import blob_doh
from skimage import io
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
import networkx as nx
from networkx2swc import networkx_to_swc
from get_tmd import get_features


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
    # Connectivity of 2 because we skeletonized
    graph, nodes = pixel_graph(image, connectivity=2)
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


def plot_graph_on_image(output_folder, name, image, centers, graph):
    """
    Plots the Minimum Spanning Tree (MST) graph on top of the original image.
    Input:
        - image: Binary image of the neurites.
        - centers: List of positions for each node, in the order of the node number.
        - graph: Minimum Spanning Tree (MST) to display.
    """
    # Swap x and y for correct positions with matplotlib
    pos = {i: (y, x) for i, (x, y) in enumerate(centers)}

    plt.figure()
    plt.imshow(image, cmap="gray")
    nx.draw(graph, pos, node_color="g", node_size=10, edge_color="r")
    plt.savefig(os.path.join(output_folder, name+'-graph.png'), bbox_inches='tight')
    plt.clf()

def closest_true_pixels(image, positions):
    """
    Computes the closest white (True) pixel for each given position in a binary image.
    Input:
        - image: 2D numpy array of True/False.
        - positions: List of (row, column) positions.
    Output:
        - closest_pixels: List of (row, column) positions corresponding to the closest white pixel for each input position.
    """
    white_pixels = np.argwhere(image)
    tree = cKDTree(white_pixels)

    # Query the tree for the closest white pixel for each input position
    closest_indices = tree.query(positions)[1]
    closest_pixels = [tuple(white_pixels[i]) for i in closest_indices]

    assert len(closest_pixels) == len(positions)
    assert all(image[pos] for pos in closest_pixels)
    return closest_pixels

def extract_label_from_filename(filename):
    # Define a regex pattern to match the desired label
    pattern = r'Canton S(.*?)(?=Mitotracker)'

    # Use re.search to find the match in the filename
    match = re.search(pattern, filename)

    if match:
        # Extract the matched label group
        info = match.group(1).strip()
        return info
    else:
        return None

def create_label_map(folder_path):
    labels = []
    label_map = {}
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".png")]

    for filename in filenames:
        label = extract_label_from_filename(filename)
        if label is not None and label not in label_map:
            labels.append(label)
            label_map[label] = len(label_map)

    return labels, label_map

def preprocess(output_folder, name, label_map, nuclei_img, dendrites_img, pers_resolution=100, graphical=False):
    """
    Pre-processes the nuclei and dendrites images (in their binary array format)
    and returns the corresponding topological features.
    """
    # Find blobs
    print("Finding the nuclei...")
    blobs = blob_doh(nuclei_img, min_sigma=30, max_sigma=80)
    neuron_centers = [(int(x), int(y)) for (x, y, _) in blobs]

    print("Refining the image...")
    # Merge the nuclei into the image since they provide useful information
    # and morph into a binary image
    merged_img = (nuclei_img != 0) | (dendrites_img != 0)
    # Binary dilation to "bone-ify" the image, hence allowing some black pixels
    # inside the paths between two nuclei
    merged_img = binary_dilation(merged_img, iterations=11)
    # Skeletonize the image to simplify shortest path finding in the pixel graph
    merged_img = skeletonize(merged_img)
    # Because we skeletonize the image, we need to find our new centers, they must be on white pixels
    # Find the closest ones to the original centers
    refined_neuron_centers = closest_true_pixels(merged_img, neuron_centers)

    print("Creating the graph...")
    graph = create_graph(merged_img, refined_neuron_centers)

    if graphical:
        # Plot the resulting graph
        plot_graph_on_image(output_folder, name, dendrites_img, neuron_centers, graph)


    print("Extracting the topological features...")
    # Extract the largest component to avoid the isolated nuclei
    largest_cc = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    networkx_to_swc(largest_cc, refined_neuron_centers, 'temp.swc')

    # Find the final x and y
    x = get_features('temp.swc', pers_resolution)
    y = label_map[extract_label_from_filename(name)]
    return np.concatenate(([y], x, [largest_cc.order()]))

def preprocess_folder(nuclei_path, dendrites_path, pers_resolution=100):
    # Find output path and create it if necessary
    output_path = os.path.join(os.path.dirname(nuclei_path), "output")
    
    # Delete the entire output directory if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Recreate the output directory
    os.makedirs(output_path)

    suffix1 = "_w3confDAPI.STK_processed.png"
    suffix2 = "_w2confmCherry.STK_processed.png"
    all_nuclei = [f for f in os.listdir(nuclei_path) if f.endswith(suffix1)]

    names = set(f.replace(suffix1, "") for f in all_nuclei)

    labels, label_map = create_label_map(nuclei_path)

    with open(os.path.join(output_path, "labels.txt"), 'w') as file:
        for i, label in enumerate(labels):
            file.write(f"{i} {label}\n")

    with open(os.path.join(output_path, "dataset.csv"), 'w') as file:
        for name in names:
            print(f"* Computing for {name}")
            nuclei_img = io.imread(os.path.join(nuclei_path, f"{name}{suffix1}"))
            dendrites_img = io.imread(os.path.join(dendrites_path, f"{name}{suffix2}"))
            try:
                res = preprocess(output_path, name, label_map, nuclei_img, dendrites_img, pers_resolution=pers_resolution, graphical=True)
                np.savetxt(file, [res], delimiter=',')
            except (np.linalg.LinAlgError, ValueError):
                print(":( Failed for this image")

if len(sys.argv) != 3:
    print("Usage: python preprocessing.py [nuclei_folder_path] [dendrites_folder_path]")
    sys.exit(1)

# Extract the two arguments
preprocess_folder(sys.argv[1], sys.argv[2])