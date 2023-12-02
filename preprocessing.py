import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.feature import blob_doh
from skimage import io
from queue import Queue
import networkx as nx

def create_adjacency_matrix(image, centers):
    """
    Creates the adjancy matrix based on the modified image in which we spilled colors from the centers of the blob.
    Two blobs will be adjacent in the resulting graph iff there are adjacent pixels of their resp. colors in the image.
    Input:
        - the binary image that has already been spilled with colors
        - the list of centers of the blobs detected in the image
    Output:
        - adjacency matrix, where 1 indicate neighbors and 0 not neighbors.
    """
    num_nodes = len(centers)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i, center in enumerate(centers):
        current_label = i + 1
        blob_pixels = np.argwhere(image == current_label)

        for pixel in blob_pixels:
            neighbors = get_adjacent_pixels(pixel, image.shape)
            for neighbor in neighbors:
                if image[neighbor] != 0 and image[neighbor] != current_label and image[neighbor] <= num_nodes:
                    adjacency_matrix[i, image[neighbor] - 1] = 1

    return adjacency_matrix

def get_adjacent_pixels(pixel, image_shape):
    """
    Helper function that returns the adjacent pixels of a pixel in a image, given its shape.
    Input:
        - (x,y) coordinates of the pixel
        - image shape
    Output:
        - list of (x,y) coordinates of neighbors
    """
    i, j = pixel
    neighbors = [
        (i + 1, j),
        (i - 1, j),
        (i, j + 1),
        (i, j - 1)
    ]

    valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < image_shape[0] and 0 <= y < image_shape[1]]
    return valid_neighbors

def initialize_blobs(image, centers):
    """
    Puts the color of each blob at the center of the blob in the image
    """
    for k, center in enumerate(centers, start=1):
        rr, cc = disk(center, 5)
        image[rr, cc] = k

def draw_circle(cx, cy, radius, image_shape):
    """
    Returns the array view for a circle centered in (cx, cy) and of radius `radius`.
    Uses the image shape.
    """
    rr, cc = np.meshgrid(
        np.arange(max(0, cy - radius), min(image_shape[0], cy + radius + 1)),
        np.arange(max(0, cx - radius), min(image_shape[1], cx + radius + 1)),
        indexing='ij'
    )
    return rr.astype(int), cc.astype(int)

def grow_blobs(image, centers):
    """
    Assuming that the centers have been initialized with their respective colors,
    spills the colors from the centers outwards, in turn, for each blob.
    Continue until no pixel can be spilled anymore from any of its direct neighbors.
    
    Modifies the image in place
    """
    queues = [Queue() for _ in range(len(centers))]
    for i, center in enumerate(centers):
        queues[i].put(tuple(center))
    while any(not queue.empty() for queue in queues):
        for k, center in enumerate(centers):
            if queues[k].empty():
                continue

            current = queues[k].get()
            i, j = current
            if 0 <= i < image.shape[0] and 0 <= j < image.shape[1] and image[i][j] == 255:
                image[i][j] = k + 1
                queues[k].put((i+1, j))
                queues[k].put((i-1, j))
                queues[k].put((i, j+1))
                queues[k].put((i, j-1))


def plot_graph_on_image(image, centers, G):
    pos = {i: (y, x) for i, (x, y) in enumerate(centers)}  # Swap x and y for correct positions

    plt.imshow(image, cmap='gray')
    nx.draw(G, pos, node_color='g', node_size=100, edge_color='r')
    plt.show()

def preprocess(nuclei_img, dendrites_img, graphical=False):
    """
    Preprocesses the nuclei and dendrites image (in their binary array format)
    and returns the corersponding features
    """
    # Find blobs
    blobs = blob_doh(nuclei_img, min_sigma=30, max_sigma=80)

    # Color them with the starting color
    for k, blob in enumerate(blobs):
        y, x, r = blob
        dendrites_img[int(x)][int(y)] = k + 1  # 255 corresponds to white in uint8

    """
    if graphical:
        # Show blobs
        fig, ax = plt.subplots()
        ax.imshow(dendrites_img, cmap='gray')
        for blob in blobs:
            x, y, r = blob
            ax.add_patch(plt.Circle((y, x), 2*r, color='r', fill=False))

        plt.show()
    """

    neuron_centers = np.array([[x,y] for (x, y, area) in blobs]).astype(int)

    # Grow the blobs until they reach black pixels
    grow_blobs(dendrites_img, neuron_centers)

    """
    if graphical:
        # Plot the result
        plt.imshow(dendrites_img, cmap='jet')
        plt.show()
    """

    adjacency_matrix = create_adjacency_matrix(dendrites_img, neuron_centers)

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Revert the image back to normal colors
    dendrites_img[dendrites_img != 0] = 255

    if graphical:
        # Plot the resulting graph
        plot_graph_on_image(dendrites_img, neuron_centers, G)
    
    # TODO: extract topological features


preprocess(io.imread('nuclei.tif'), io.imread('dendrites.tif'), graphical=True)
