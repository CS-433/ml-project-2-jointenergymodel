"""
Computing the Topological Morphology Descriptor (TMD) and other topological features
"""

import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gudhi.representations import PersistenceImage
from scipy import stats

def compute_tmd(tree, positions):
    """
    tree is the tree structure, is Networkx
    positions is a dictionary with the positions (numpy array) associated to each node
    """
    assert nx.is_tree(tree)
    assert tree.size() > 0
    N = len(positions)

    # get root
    node_sequence = sorted(tree.degree, key=lambda x: x[1], reverse=True)
    root = node_sequence[0][0]

    # construct leaves
    Leaves = []
    for node in node_sequence:
        if (node[1] == 1) and (node[0] != root):
            Leaves.append(node[0])

    # construct parents and children
    parent = np.zeros(N, dtype=int) - 1
    children = [[] for i in range(N)]
    for parent_id, child_id in nx.dfs_edges(tree, source=root):
        parent[child_id] = parent_id
        children[parent_id].append(child_id)

    TMD = []
    # A is the list of active nodes
    A = Leaves
    v = np.zeros(N)
    for leaf in Leaves:
        v[leaf] = nx.shortest_path_length(tree, source=root, target=leaf)
    while not (root in A):
        for leaf in A:
            p = parent[leaf]
            C = children[p]

            CinA = True
            for n in C:
                CinA = (n in A) and CinA
            if CinA:
                cm = C[0]
                max_vc = v[cm]
                for c in C:
                    if v[c] > max_vc:
                        cm = c
                        max_vc = v[c]
                A.append(p)
                for ci in C:
                    A.remove(ci)
                    if ci != cm:
                        TMD.append((v[ci], nx.shortest_path_length(tree, source=root, target=p)))
                v[p] = v[cm]
    TMD.append((v[root], 0))
    return np.array(TMD)


def get_limits(phs_list):
    """Returns the x-y coordinates limits (min, max) for a list of persistence diagrams."""
    ph = copy.deepcopy(phs_list)
    xlim = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    ylim = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]
    return xlim, ylim

def get_tmd_vector_depr(bc, reso=100, graphic=False):
    """
    Compute the flatten persistence image associated with the barcode bc using the method from TMD library
    """

    xlim, ylim = get_limits(bc)
    res = complex(0, reso)
    X, Y = np.mgrid[xlim[0] : xlim[1] : res, ylim[0] : ylim[1] : res]

    values = np.transpose(bc).astype(np.float64)
    if values.shape[1] == 1:
        values = np.concatenate((values, np.array([2, 0]).reshape((2, 1))), axis=1)
    offset = np.random.rand(values.shape[1])
    values[1] += offset
    try:
        kernel = stats.gaussian_kde(values, bw_method=None, weights=None)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
    except:
        Z = np.ones((100, 100))

    norm_factor = np.max(Z)

    if graphic:
        plt.imshow(Z / norm_factor)
        plt.title("Persistence Image")
        plt.show()

    return (Z / norm_factor).flatten()

def get_tmd_vector(bc, reso=100):
    """
    Compute the flatten persistence image associated with the barcode bc using GUDHI library
    """
    n = bc.shape[0]
    PI = PersistenceImage(bandwidth=n**(-1.0/6), resolution=[reso, reso])
    pi = PI.fit_transform([bc])

    pi2 = np.flip(np.reshape(pi[0], [reso, reso]), 0)
    norm_factor = np.max(pi2)
    pi2 = pi2/norm_factor
    return pi2.flatten()

def get_persistent_entropy(ph_neu):
    """
    Computes the persistent entropy of a persistence diagram.

    Parameters:
    - ph_neu: Persistence diagram.

    Returns:
    - float: Persistent entropy value.
    """
    # Extract the persistence intervals from the input
    intervals = ph_neu

    # Filter out infinite intervals (unbounded persistence)
    finite_intervals = np.array(
        [interval for interval in intervals if np.isfinite(interval).all()]
    )

    # Handle the case where there are no finite intervals
    if len(finite_intervals) == 0:
        return 0.0

    # Calculate the length of each bar
    lengths = np.abs(finite_intervals[:, 1] - finite_intervals[:, 0])

    # Calculate the total length of all bars
    total_length = np.sum(lengths)

    if total_length == 0:
        return 0.0

    # Compute the persistent entropy using the formula from the paper
    persistent_entropy = -np.sum(
        lengths / total_length * np.log((lengths + 1e-10) / total_length)
    )

    return persistent_entropy


def get_features(graph, pos, resolution):
    # In case the graph is a point
    if graph.size() == 0:
        barcode = np.array([[1, 0], [2, 0]])
        persistent_entropy = 0
        # image = get_tmd_vector(barcode, resolution)
        image = np.zeros((resolution,resolution))
        return np.concatenate((image, [persistent_entropy]))

    # Compute the barcode
    barcode = compute_tmd(graph, np.array(pos))

    persistent_entropy = get_persistent_entropy(barcode)

    # Get the persistance image
    image = get_tmd_vector(barcode, resolution)

    return np.concatenate((image, [persistent_entropy]))
