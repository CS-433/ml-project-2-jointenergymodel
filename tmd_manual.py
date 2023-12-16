import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gudhi

def compute_tmd(root, tree, positions, Leaves):
    """
    root is the index of the root in tree
    tree is the tree structure, is Networkx
    positions is a N*d array with the positions
    Leaves is the list of leaves
    """
    assert nx.is_tree(tree)
    N = tree.size() + 1   #number of nodes

    #construct parents and children
    parent = np.zeros(N, dtype=int) - 1
    children = [[] for i in range(N)]
    for parent_id, child_id in nx.dfs_edges(tree, source=root):
        print(parent_id)
        parent[child_id] = parent_id
        children[parent_id].append(child_id)

    print(parent)
    TMD = []
    #A is the list of active nodes
    A = Leaves
    v = np.zeros(N)
    for leaf in Leaves:
        v[leaf] = np.linalg.norm(positions[leaf]-positions[root])
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
                        TMD.append((v[ci], np.linalg.norm(positions[p]-positions[root])))
                v[p] = v[cm]
    TMD.append((v[root], 0))
    return TMD


graph = nx.generators.trees.random_tree(10)
pos = nx.spring_layout(graph)

node_sequence = sorted(graph.degree, key=lambda x: x[1], reverse=True)
soma = node_sequence[0][0]

L = []
for node in node_sequence:
    if node[1] == 1:
        L.append(node[0])

barcode = compute_tmd(soma, graph, pos, L)

print(barcode)
nx.draw(graph, pos=pos, with_labels=True)
diag = np.array(barcode)
print(diag)

#fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
gudhi.plot_persistence_diagram(persistence=diag)
gudhi.plot_persistence_barcode(persistence=diag)


print(gudhi.__version__)
PI = gudhi.representations.PersistenceImage(bandwidth=1e-4, weight=lambda x: x[1]**2, im_range=[0,.004,0,.004], resolution=[100,100])
pi = PI.fit_transform([diag])

plt.imshow(np.flip(np.reshape(pi[0], [100,100]), 0))
plt.title("Persistence Image")





plt.show()