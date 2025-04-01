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

from scipy.ndimage import _nd_image
from scipy.ndimage import spline_filter
from scipy.ndimage import _ni_support

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


def optimized_map_coordinates(input, coordinates, output=None, order=3,
                              mode='constant', cval=0.0, prefilter=True):
    """
    Optimized version of map_coordinates that skips unnecessary computations.

    Parameters
    ----------
    %(input)s
    coordinates : array_like
        The coordinates at which `input` is evaluated.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s

    Returns
    -------
    map_coordinates : ndarray
        The result of transforming the input. The shape of the output is
        derived from that of `coordinates` by dropping the first axis.
    """
    # Early exit if order is out of range
    if order < 0 or order > 5:
        raise ValueError('Spline order not supported')

    input = np.asarray(input)
    coordinates = np.asarray(coordinates)

    # Early exit if coordinates are complex (if your use case doesn't need this)
    if np.iscomplexobj(coordinates):
        raise TypeError('Complex type not supported')

    # Check if input and coordinates have compatible shapes
    output_shape = coordinates.shape[1:]
    if input.ndim < 1 or len(output_shape) < 1:
        raise ValueError('Input and output rank must be > 0')

    if coordinates.shape[0] != input.ndim:
        raise ValueError('Invalid shape for coordinate array')

    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)

    # Handle complex input by splitting real and imaginary parts (if needed)
    if complex_output:
        kwargs = dict(order=order, mode=mode, prefilter=prefilter)
        optimized_map_coordinates(input.real, coordinates, output=output.real,
                                  cval=np.real(cval), **kwargs)
        optimized_map_coordinates(input.imag, coordinates, output=output.imag,
                                  cval=np.imag(cval), **kwargs)
        return output

    # Skipping prefiltering for order <= 1
    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, output=np.float64,
                                 mode=mode)
    else:
        npad = 0
        filtered = input

    mode = _ni_support._extend_mode_to_code(mode)

    # Optimized geometric transform operation
    _nd_image.geometric_transform(filtered, None, coordinates, None, None,
                                  output, order, mode, cval, npad, None, None)

    return output


def unwarp_image_backward_tester(mat, indices, order=1, mode="reflect"):
    """
    Unwarp an image using the previously computed indices and apply transformation.

    Parameters
    ----------
    mat : array_like
        2D array.
    indices : tuple
        A tuple containing the transformed y and x indices for the mapping.
    order : int, optional
        The order of the spline interpolation.
    mode : {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest',
           'mirror', 'grid-wrap', 'wrap'}, optional
        To determine how to handle image boundaries.

    Returns
    -------
    array_like
        2D array. Distortion-corrected image.
    """
    # Apply the transformation using map_coordinates
    mat_transformed = map_coordinates(mat, indices, order=order, mode=mode)
    #mat_transformed = optimized_map_coordinates(mat, indices, order=order, mode=mode)

    # Get the original shape of the input image (mat)
    height, width = mat.shape
    
    # Ensure the result is reshaped to match the original image shape
    return mat_transformed.reshape((height, width))  # Reshape to original dimensions

def compute_indices(xcenter, ycenter, list_fact, mat):
    """
    Compute the transformation grid indices that can be reused across multiple calls.
    
    Parameters
    ----------
    xcenter : float
        Center of distortion in x-direction.
    ycenter : float
        Center of distortion in y-direction.
    list_fact : list of float
        Polynomial coefficients of the backward model.
    mat : array_like
        2D array (image matrix).

    Returns
    -------
    indices : tuple
        A tuple containing the transformed y and x indices for the mapping.
    """
    (height, width) = mat.shape
    xu_list = np.arange(width) - xcenter
    yu_list = np.arange(height) - ycenter
    xu_mat, yu_mat = np.meshgrid(xu_list, yu_list)
    ru_mat = np.sqrt(xu_mat ** 2 + yu_mat ** 2)
    fact_mat = np.sum(np.asarray(
        [factor * ru_mat ** i for i, factor in enumerate(list_fact)]), axis=0)
    xd_mat = np.float32(np.clip(xcenter + fact_mat * xu_mat, 0, width - 1))
    yd_mat = np.float32(np.clip(ycenter + fact_mat * yu_mat, 0, height - 1))
    indices = np.reshape(yd_mat, (-1, 1)), np.reshape(xd_mat, (-1, 1))

    return indices