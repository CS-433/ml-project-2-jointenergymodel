import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.feature import blob_log
from skimage import io
from queue import Queue

def initialize_blobs(image, centers):
    for k, center in enumerate(centers, start=1):
        rr, cc = disk(center, 1)
        image[rr, cc] = k

def draw_circle(cx, cy, radius, image_shape):
    rr, cc = np.meshgrid(
        np.arange(max(0, cy - radius), min(image_shape[0], cy + radius + 1)),
        np.arange(max(0, cx - radius), min(image_shape[1], cx + radius + 1)),
        indexing='ij'
    )
    return rr.astype(int), cc.astype(int)

def grow_blobs(image, centers):
    queues = [Queue() for _ in range(len(centers))]
    for i, center in enumerate(centers):
        queues[i].put(tuple(center))
    while any(not queue.empty() for queue in queues):
        for k, center in enumerate(centers):
            if queues[k].empty():
                continue

            current = queues[k].get()
            i, j = current
            if 0 <= i < image.shape[0] and 0 <= j < image.shape[1] and image[i][j] == 255:
                image[i][j] = k + 1
                queues[k].put((i+1, j))
                queues[k].put((i-1, j))
                queues[k].put((i, j+1))
                queues[k].put((i, j-1))

def plot_image_with_labels(image):
    plt.imshow(image, cmap='jet')
    plt.show()


nuclei_img = io.imread('nuclei.tif')
dendrites_img = io.imread('dendrites.tif')

# Find blobs
blobs = blob_log(nuclei_img, max_sigma=80, min_sigma = 15, threshold=0.3, overlap = 0.2)

# Color them with the starting color
for k, blob in enumerate(blobs):
    y, x, r = blob
    dendrites_img[int(x)][int(y)] = k + 1  # 255 corresponds to white in uint8

# Show blobs
fig, ax = plt.subplots()
ax.imshow(dendrites_img, cmap='gray')
for blob in blobs:
    y, x, area = blob
    ax.add_patch(plt.Circle((x, y), area*np.sqrt(2), color='r', fill=False))

plt.show()

neuron_centers = np.array([[x,y] for (x, y, area) in blobs]).astype(int)

# Grow the blobs until they reach black pixels
grow_blobs(dendrites_img, neuron_centers)

# Plot the result
plot_image_with_labels(dendrites_img)
