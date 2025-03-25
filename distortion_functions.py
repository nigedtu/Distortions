import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.io import savemat
import cv2
from scipy.optimize import least_squares
from skimage.measure import label, regionprops
from skimage.transform import radon
from skimage.feature import corner_harris, corner_peaks
from preprocessing import *
from processing import *
from postprocessing import *
from linepattern import *
import loadersaver as losa
from scipy.ndimage import center_of_mass, label
import itertools 
from datetime import datetime
import time
import timeit
import winsound



from skimage.measure import regionprops
import scipy.ndimage as ndi
from skimage.segmentation import clear_border

def compute_centroids_from_matrix2(centroid_matrix):
    # Label connected components
    num_labels, labels = cv2.connectedComponents(centroid_matrix.astype(np.uint8))

    # Get all coordinates of non-zero pixels once
    coords = np.argwhere(labels > 0)

    # Use a dictionary to collect coordinates per label
    from collections import defaultdict
    label_coords = defaultdict(list)

    for y, x in coords:
        label = labels[y, x]
        label_coords[label].append((y, x))

    # Compute centroids
    centroids = []
    for label, points in label_coords.items():
        points = np.array(points)
        centroid_y = np.mean(points[:, 0])
        centroid_x = np.mean(points[:, 1])
        centroids.append((centroid_y, centroid_x))

    return centroids

def harris_corners_to_centroid_matrix2(image, square_size=26, corner_threshold_factor=0.022):
    """
    Detects corner intersections using Harris corner detection and returns a matrix with detected corners marked as 1.
    
    Parameters:
    - image: The input image (should be grayscale).
    - square_size: The size of each square in the checkerboard pattern (default 26).
    - corner_threshold_factor: A factor to scale the corner detection threshold.
    
    Returns:
    - output_matrix: A binary matrix with 1s marking detected corners.
    """
    start_time = time.time()
    image_float = np.float32(image)
    
    # Adjust blockSize based on square size (larger blocks for larger squares)
    block_size = int(square_size / 2)  # Set the blockSize as half of the square size.
    
    # Perform Harris corner detection
    corners = cv2.cornerHarris(image_float, blockSize=block_size, ksize=3, k=0.04)
    
    # Dilate to enhance corner regions
    corners_dilated = cv2.dilate(corners, None)
    
    # Apply threshold to detect strong corners
    corner_thresh = corner_threshold_factor * corners_dilated.max()
    
    # Create output matrix with 1s marking detected corners
    output_matrix = np.zeros_like(image, dtype=np.uint8)  # Initialize as 0s
    output_matrix[corners_dilated > corner_thresh] = 1  # Mark detected corners as 1
    
    return output_matrix  # Returns binary matrix with detected corners




