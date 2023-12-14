import tmd

def get_features(path, resolution):
    """
    Extract features of the input graph
    Input:
        - path: A string of the path to the .swc
        - visual: A Bool that indicates if the user wants visuals
    """
    # Load the neuron
    neu = tmd.io.load_neuron(path)

    # Extract the ph diagram of a neuron's trees
    ph_neu = tmd.methods.get_ph_neuron(neu)

    # Get the persistance image
    image = tmd.analysis.get_persistence_image_data(ph_neu, resolution=resolution)

    return image.flatten()