import cv2
import numpy as np

def locate_nuclei(image_path, merge_radius=5):
    """
    Finds the nuclei in the binary image given by `image_path`.
    Thanks ChatGPT.
    
    Input:
    - image_path: relative or full path to the binary image
    - merge_radius: distance between two blogs that can be merged into one nucleus
    
    Output:
    - nuclei_coordinates: returns a list of (x,y) coordinates of the nuclei in the picture.
    """
    # Read the binary image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Find connected components in the binary image
    _, labels = cv2.connectedComponents(img)

    # Merge closely located nuclei using a distance threshold
    merged_labels = np.zeros_like(labels)
    nuclei_coordinates = []

    for label in range(1, np.max(labels) + 1):
        # Extract the region corresponding to the current label
        mask = np.uint8(labels == label)

        # Find the centroid of the region
        m = cv2.moments(mask)
        if m["m00"] != 0:
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])
            centroid = (cX, cY)

            # Check if there is a closely located nucleus
            merge = False
            for existing_centroid in nuclei_coordinates:
                dist = np.linalg.norm(np.array(centroid) - np.array(existing_centroid))
                if dist < merge_radius:
                    merge = True
                    break

            if merge:
                # Merge the centroids
                existing_centroid = np.array(existing_centroid)
                centroid = np.array(centroid)
                new_centroid = tuple(np.round((existing_centroid + centroid) / 2).astype(int))
                nuclei_coordinates = [c for c in nuclei_coordinates if not np.all(c == existing_centroid)]
                nuclei_coordinates.append(new_centroid)
            else:
                # Add as a new nucleus
                nuclei_coordinates.append(centroid)

    return nuclei_coordinates
