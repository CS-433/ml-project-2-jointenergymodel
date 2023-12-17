import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gudhi import plot_persistence_diagram, plot_persistence_barcode
from gudhi.representations import PersistenceImage


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


def get_TMD_vector(bc, reso=100, graphic=False):
    """
    compute the flatten persistence image associted to the barcode bc
    """
    PI = PersistenceImage(bandwidth=1, resolution=[reso, reso])
    pi = PI.fit_transform([bc])

    pi2 = np.flip(np.reshape(pi[0], [reso, reso]), 0)
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
