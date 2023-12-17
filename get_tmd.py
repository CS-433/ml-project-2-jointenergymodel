import numpy as np
from tmd_manual import compute_tmd, get_TMD_vector


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

def get_features(graph, pos, resolution):
    # In case the graph is a point
    if graph.size() == 0:
        barcode = np.array([[0,0]])
        persistent_entropy = 0
        image = get_TMD_vector(barcode, resolution, False)
        return np.concatenate((image, [persistent_entropy]))


    # Compute the barcode
    barcode = compute_tmd(graph, np.array(pos))

    persistent_entropy = get_persistent_entropy(barcode)

    # Get the persistance image
    image = get_TMD_vector(barcode, resolution, False)

    return np.concatenate((image, [persistent_entropy]))