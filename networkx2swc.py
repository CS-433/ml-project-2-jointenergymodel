import networkx as nx

def networkx_to_swc(graph, neurons, output_file):
    """
    Writes an swc file of the Networkx tree in input.
    Input:
        - graph: a Networkx graph that has to be a tree
        - neurons a List of (row, column) positions corresponding to the positions of each nodes
    Writes:
        - an SWC file
    """
    with open(output_file, 'w') as f:
        assert nx.is_tree(graph)

        node_id_mapping = {}
        current_id = 1
        node_sequence = sorted(graph.degree, key=lambda x: x[1], reverse=True)

        # Write the root node (assuming it is node 1)
        soma = node_sequence[0][0]
        x, y = neurons[soma]
        f.write(f"1 1 {x} {y} 0 2 -1\n")
        f.write(f"2 3 {x} {y} 0 2 1\n")
        node_id_mapping[soma] = 2
        current_id += 2


        for parent_id, child_id in nx.dfs_edges(graph, source=soma):
            parent_id = node_id_mapping[parent_id]
            x, y = neurons[child_id]

            f.write(f"{current_id} 3 {x} {y} 0 1 {parent_id}\n")

            node_id_mapping[child_id] = current_id
            current_id += 1
