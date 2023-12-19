import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gudhi import plot_persistence_diagram, plot_persistence_barcode
from gudhi.representations import PersistenceImage

import copy
import math
from itertools import chain

import numpy as np
from numpy.linalg import norm
from scipy import stats
from scipy.spatial.distance import cdist


def compute_tmd(tree, positions):
    """
    tree is the tree structure, is Networkx
    positions is a dictionary with the positions (numpy array) associated to each node
    """
    assert nx.is_tree(tree)
    assert (tree.size() > 0)
    N = len(positions)

    # get root
    node_sequence = sorted(tree.degree, key=lambda x: x[1], reverse=True)
    root = node_sequence[0][0]

    # construct leaves
    Leaves = []
    for node in node_sequence:
        if node[1] == 1:
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
        v[leaf] = np.linalg.norm(positions[leaf] - positions[root])
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
                        TMD.append(
                            (v[ci], np.linalg.norm(positions[p] - positions[root]))
                        )
                v[p] = v[cm]
    TMD.append((v[root], 0))
    return np.array(TMD)

def get_limits(phs_list):
    """Returns the x-y coordinates limits (min, max) for a list of persistence diagrams."""
    ph = copy.deepcopy(phs_list)
    xlim = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    ylim = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]
    return xlim, ylim

def get_TMD_vector(bc, reso=100, graphic=False):
    """
    compute the flatten persistence image associted to the barcode bc
    """
    # PI = PersistenceImage(bandwidth=100, resolution=[reso, reso])
    # pi = PI.fit_transform([bc])

    # pi2 = np.flip(np.reshape(pi[0], [reso, reso]), 0)


    xlim, ylim = get_limits(bc)
    res = complex(0, reso)
    X, Y = np.mgrid[xlim[0] : xlim[1] : res, ylim[0] : ylim[1] : res]

    values = np.transpose(bc)
    print(values.shape)
    kernel = stats.gaussian_kde(values, bw_method=None, weights=None)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    norm_factor = np.max(Z)

    return (Z / norm_factor).flatten()


    if graphic:
        plt.imshow(pi2)
        plt.title("Persistence Image")
        plt.show()
    return pi2.flatten()


'''
graph = nx.generators.trees.random_tree(15)
pos = nx.spring_layout(graph)

# node_sequence = sorted(graph.degree, key=lambda x: x[1], reverse=True)
# soma = node_sequence[0][0]

# L = []
# for node in node_sequence:
#     if node[1] == 1:
#         L.append(node[0])

barcode = compute_tmd(graph, pos)

print(barcode)
nx.draw(graph, pos=pos, with_labels=True)
# diag = np.array(barcode)
# print(diag)

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
# plot_persistence_diagram(persistence=diag)
# plot_persistence_barcode(persistence=diag)


# PI = PersistenceImage(resolution=[100,100])
# pi = PI.fit_transform([diag])

# plt.imshow(np.flip(np.reshape(pi[0], [100,100]), 0))
# plt.title("Persistence Image")

vect = get_TMD_vector(barcode, 20, False)
print(vect.shape)


plt.show()
'''
