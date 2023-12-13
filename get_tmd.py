# Import the TMD toolkit in IPython
import tmd
from tmd.view import view
from matplotlib import pyplot as plt

import networkx as nx

# def convert_to_swc(graph, output_file):
#     with open(output_file, 'w') as f:
#         node_id = 1
#         for node in nx.topological_sort(graph):
#             parent_id = node_id - 1 if node_id > 1 else -1  # Set -1 for the root node
#             #x, y, z = graph.nodes[node]['pos']  # Assuming you have 'pos' attribute for node positions
#             #radius = graph.nodes[node].get('radius', 1.0)  # Default radius is 1.0 if not specified

#             line = f"{node_id} 1 x y z 0.1 {parent_id}\n"
#             f.write(line)

#             node_id += 1


# def convert():
#     # Create a sample graph using NetworkX
#     G = nx.path_graph(10)
#     nx.draw(G)
#     pos = nx.spring_layout(G)
#     nx.set_node_attributes(G, pos, 'pos')

#     # Convert and save to SWC file
#     convert_to_swc(G, 'output.swc')

# convert()


def get_features(path, resolution):
    """
    Extract features of the input graph
    Input:
        - path: A string of the path to the .swc
        - visual: A Bool that indicates if the user wants visuals
    """
    # Load a neuron
    neu = tmd.io.load_neuron(path)

    # Visualize the neuron
    # view.neuron(neu)
    # plt.show()

    # Extract the tmd of a neurite, i.e., neuronal tree
    ph = tmd.methods.get_persistence_diagram(neu.neurites[0])


    # Step 4: Extract the ph diagram of a neuron's trees
    ph_neu = tmd.methods.get_ph_neuron(neu)
    # print(ph_neu)

    # Step 5: Extract the ph diagram of a neuron's trees,
    # depending on the neurite_type
    # ph_apical = tmd.methods.get_ph_neuron(neu, neurite_type="apical_dendrite")
    # ph_axon = tmd.methods.get_ph_neuron(neu, neurite_type="axon")
    # ph_basal = tmd.methods.get_ph_neuron(neu, neurite_type="basal_dendrite")

    # Step 6: Plot the extracted topological data with three different ways
    pers_image2test = tmd.analysis.get_persistence_image_data(ph_neu, resolution=resolution)
    print(pers_image2test)
    
    # plot.diagram(ph_neu)

    # # Visualize a selected neurite type or multiple of them
    # view.neuron(neu, neurite_type=["apical_dendrite"])

    # # Visualize the persistence diagram
    # plot.diagram(ph)

    # # Visualize the persistence barcode
    # plot.barcode(ph)

    # # Visualize the persistence image
    # plt.show()
    # plot.persistence_image(ph)

    plt.show()
    # print(Zn, cmtruc)
    plt.imshow(pers_image2test)

    # plt.show()
    return pers_image2test.flatten()