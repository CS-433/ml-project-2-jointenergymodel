"""
Data pre-processing
Converts binary images of nuclei and dendrites into a Minimum Spanning Tree (MST) graph representation.

The main function is `preprocess`, which takes both binary image channels and returns the corresponding topological features.
Then `preprocess_folder` executes it on a batch of files.
"""

import sys
import os
import shutil
import multiprocessing as mp
import urllib.request
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import pixel_graph
from skimage.feature import blob_doh
from skimage import io
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
import networkx as nx
from sympy import symbols, factor_list, Poly
from tmd import get_features
from config import INPUT_PATH


def load_input_folder():
    """Retrieves the input folder from the cloud, if necessary"""
    # Check if the input folder already exists
    if os.path.exists("input"):
        return

    print("Downloading the dataset...")
    # Download the zip file
    urllib.request.urlretrieve(INPUT_PATH, "input.zip")

    # Unzip the file
    print("Unzipping the dataset...")
    with zipfile.ZipFile("input.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    # Remove the downloaded zip file
    os.remove("input.zip")
    print("Dataset is downloaded")


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

    return graph, mst


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

    plt.figure()
    plt.imshow(image, cmap="gray")
    nx.draw(graph, pos, node_color="g", node_size=80, width=2.0, edge_color="r")
    plt.show()


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


def graph_polynomial_factorization_coefficients(graph):
    x = symbols("x")
    polynomial = sum(x ** graph.degree(node) for node in graph.nodes)

    _, factors = factor_list(polynomial)
    factors = [factor[0] for factor in factors]
    factors = [Poly(factor).all_coeffs() for factor in factors]
    N = 10
    K = 8
    factors = np.array(
        [factor[-N:] + [0] * max(0, N - len(factor)) for factor in factors[:K]]
        + [[0] * N] * max(0, K - len(factors))
    )
    return factors


def preprocess(
    output_folder,
    name,
    label,
    nuclei_img,
    dendrites_img,
    pers_resolution,
    graphical=False,
):
    """
    Pre-processes the nuclei and dendrites images (in their binary array format)
    and returns the corresponding topological features.
    """
    # Find blobs
    print(f"[{name}] Finding the nuclei...")
    blobs = blob_doh(nuclei_img, min_sigma=10, max_sigma=80)
    neuron_centers = [(int(x), int(y)) for (x, y, _) in blobs]

    if graphical:
        # Show blobs
        fig, ax = plt.subplots()
        ax.imshow(nuclei_img, cmap="gray")
        for blob in blobs:
            x, y, r = blob
            ax.add_patch(plt.Circle((y, x), r, color="g", fill=False, linewidth=2.0))

        plt.show()

    print(f"[{name}] Refining the image...")
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

    if graphical:
        # Show blobs
        fig, ax = plt.subplots()
        ax.imshow(merged_img, cmap="gray")
        plt.show()

    print(f"[{name}] Creating the graph...")
    graph0, graph = create_graph(merged_img, refined_neuron_centers)

    if graphical:
        # Plot the graph before being a tree
        plot_graph_on_image(dendrites_img, neuron_centers, graph0)
        # Plot the resulting graph
        plot_graph_on_image(dendrites_img, neuron_centers, graph)

    print(f"[{name}] Extracting the topological features...")
    # Rename the nodes
    assert graph.order() == len(neuron_centers)
    graph = nx.relabel_nodes(
        graph, {old: new for new, old in enumerate(neuron_centers)}
    )
    # Extract the largest component to avoid the isolated nuclei
    largest_cc = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

    factors = graph_polynomial_factorization_coefficients(largest_cc)

    if graphical:
        # Plot the connected component we consider
        plot_graph_on_image(dendrites_img, neuron_centers, largest_cc)

    # Find the final features
    x = get_features(largest_cc, neuron_centers, pers_resolution)

    return np.concatenate(([label], x, factors.flatten(), [largest_cc.order()]))


def atomic_preprocessing(input_args):
    """
    Preprocessing function that can actually be parallelized per image
    """
    y, label, nuclei_path, input_path, output_path, pers_resolution = input_args
    name = nuclei_path.replace("w3confDAPI", "")
    dendrites_path = nuclei_path.replace("w3confDAPI", "w2confmCherry")
    nuclei_img = io.imread(os.path.join(input_path, label, "nuclei", nuclei_path))
    dendrites_img = io.imread(
        os.path.join(input_path, label, "dendrites", dendrites_path)
    )

    return preprocess(
        output_path,
        name,
        y,
        nuclei_img,
        dendrites_img,
        pers_resolution,
    )


def preprocess_folder(pers_resolution=100):
    # Input path is always `input`
    input_path = "input"

    # Retrieves input folder, if necessary
    load_input_folder()

    # Find output path and create it if necessary
    output_path = os.path.join(os.path.dirname(input_path), "output")

    # Delete the entire output directory if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Recreate the output directory
    os.makedirs(output_path)

    # Get all sub-folders/classification labels
    labels = sorted(
        [
            f
            for f in os.listdir(input_path)
            if os.path.isdir(os.path.join(input_path, f))
        ]
    )

    # Save the labels
    with open(os.path.join(output_path, "labels.txt"), "w") as file:
        for i, label in enumerate(labels):
            file.write(f"{i} {label}\n")

    # Compute the dataset in parallel
    nuclei_paths = [
        (y, label, path, input_path, output_path, pers_resolution)
        for y, label in enumerate(labels)
        for path in os.listdir(os.path.join(input_path, label, "nuclei"))
    ]

    # Compute all in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(atomic_preprocessing, nuclei_paths)

        # Save results to the file
        with open(os.path.join(output_path, "dataset.csv"), "w") as file:
            for res in results:
                np.savetxt(file, [res], delimiter=",")


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python preprocessing.py")
        sys.exit(1)

    # Uncomment if you want to do preprocessing on a single image
    nuclei_img = "nuclei3.png"
    dendrites_img = "dendrites3.png"
    """preprocess(
        'output',
        'example',
        0,
        io.imread(nuclei_img),
        io.imread(dendrites_img),
        100,
    )"""

    # Extract the two arguments
    preprocess_folder()
