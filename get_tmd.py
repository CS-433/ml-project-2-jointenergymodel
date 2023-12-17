import tmd
import numpy as np


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

    # Compute the persistent entropy using the formula from the paper
    persistent_entropy = -np.sum(
        lengths / total_length * np.log((lengths + 1e-10) / total_length)
    )

    return persistent_entropy


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

    persistent_entropy = get_persistent_entropy(ph_neu)

    return np.concatenate((image.flatten(), [persistent_entropy]))
