import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

def networkx_to_swc(graph, neurons, output_file):
    """
    Writes an swc file of the Networkx tree in input.
    Input:
        - graph: a Netwotkx graph that has to be a tree
        - neurons a List of (row, column) positions corresponding to the positions of each nodes
    Writes:
        - an SWC file
    """
    with open(output_file, 'w') as f:
        assert(nx.is_tree(graph))


        node_id_mapping = {}
        current_id = 1
        node_sequence = sorted(graph.degree, key=lambda x: x[1], reverse=True)
        ns = (np.array(node_sequence)[:,1])
        nb_soma = np.count_nonzero(ns==1)
        print(np.count_nonzero(ns==1))
        soma = node_sequence[0][0]
        node_id_mapping[soma] = current_id + 1



        # Write the root node (assuming it is node 1)
        x, y = neurons[soma]
        line = f"{current_id} 1 {x} {y} 0.0 0.2 -1\n"
        f.write(line)
        line = f"{current_id+1} 3 {x} {y} 1.0 0.2 1\n"
        f.write(line)
        current_id +=1
        current_id += 1
        nb_soma -= 1


        for node in nx.dfs_edges(graph, source=soma):
            parent_id, child_id = node
            
            parent_data = graph.nodes[parent_id]
            child_data = graph.nodes[child_id]

            parent_id = node_id_mapping[parent_id]
            x, y = neurons[child_id]
            # radius = child_data.get('radius', 1.0)
            # neuron_type = child_data.get('type', 1)
            type_n = 3
            # if nb_soma>0:
            #     type_n = 1
            #     nb_soma -= 1

            line = f"{current_id} {type_n} {x} {y} 0.0 0.1 {parent_id}\n"
            f.write(line)

            node_id_mapping[child_id] = current_id
            current_id += 1

        

# # Exemple d'utilisation
# G = nx.Graph()
# G = nx.generators.trees.random_tree(10)
# H = nx.path_graph(10)

# # G.add_node(1, pos=(0, 0, 0), radius=1.0, type=1)
# # G.add_node(2, pos=(1, 0, 0), radius=0.8, type=2)
# # G.add_edge(1, 2)
# print(G.degree())
# print(nx.is_tree(G))
# node_sequence = sorted(G.degree, key=lambda x: x[1], reverse=True)
# soma = node_sequence[0][0]

# networkx_to_swc(G, 'output.swc')

# nx.draw(G)
# plt.show()
