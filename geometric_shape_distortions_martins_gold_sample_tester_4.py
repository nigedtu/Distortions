import os
# Change to your local directory:
os.chdir(r"C:\Users\nige\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Postdoc\Projects\Detector paper\20241204 beamtimedata\process\fields_corrected\MLL_new_detector")

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
time_start = timeit.default_timer()


'''
fnames = ['scan-2689_ximea_corrected.h5','scan-2689_pt_ximea_corrected_gradients.h5','scan-2689_ximea_corrected_gradients.h5'] # 
with h5py.File(fnames[1], 'r') as file:
    gradient_data_pt = file['gradient/data'][:] 

#fig, axs = plt.subplots(1,2,figsize=(21, 6))
#im=axs[0].imshow(gradient_data_pt[0])
#im=axs[1].imshow(gradient_data_pt[1])

with h5py.File(fnames[0], 'r') as file:
    corrected_data = file['data/corrected'][:]
    
with h5py.File(fnames[2], 'r') as file:
    gradient_data = file['gradient/data'][:]

fig, axs = plt.subplots(1,2,figsize=(21, 6))
im=axs[0].imshow(corrected_data,cmap='gray')
axs[1].imshow(corrected_data[3120:3270,3120:3270], cmap='gray')

#fig, axs = plt.subplots(1,2,figsize=(21, 6))
#im=axs[0].imshow(gradient_data[0])
#im=axs[1].imshow(gradient_data[1])
'''

# Change to your local directory:
os.chdir(r"C:\Users\nige\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Postdoc\Projects\Detector paper\20250225 beamtimedata\process\field_corrected")
fnames = ['grid_scan_2398_corrected.h5'] # Martins Bech Checkerboard patter, measured at DanMAX February 2025


with h5py.File(fnames[0], 'r') as file:
    image_data = file['corrected/image']
    corrected_data_2 = image_data[:]
    
fig, axs = plt.subplots(1,2,figsize=(21, 6))
im=axs[0].imshow(corrected_data_2,cmap='gray')
axs[1].imshow(corrected_data_2[3120:3270,3120:3270], cmap='gray')



#%%

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


# Some images are saved. Select where you want them to be saved: 
output_base = "../"
# Select data
image_data = corrected_data_2[3787:9787,3921:9921] # (10640, 13856)  # cod: (6787,6921)
image_data = corrected_data_2[0:-1,0:-1] # (10640, 13856)  # cod: (6787,6921)
image_data = np.where(np.isnan(image_data), 0.4, image_data) # Remove empty pixels
mat0 = image_data # Initiate to use Discorypy nameing        https://github.com/DiamondLightSource/discorpy/tree/master

centroid_matrix = harris_corners_to_centroid_matrix2(image_data, square_size=23, corner_threshold_factor=0.001) # Find corners using CV2 HarrisCorners. 
centroids = compute_centroids_from_matrix2(centroid_matrix) # Find Centroids. 

# Step 3: Create a new matrix with a single dot for each centroid
new_centroid_matrix = np.zeros_like(centroid_matrix)
for (cy, cx) in centroids:
    new_centroid_matrix[int(cy), int(cx)] = 1  # Mark the centroid in the new matrix

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(21, 6))
# Plot the original image
axs[0].imshow(image_data, cmap='gray')
axs[0].set_title("Original Image")

# Plot the new centroid matrix with single centroids
axs[1].imshow(new_centroid_matrix, cmap='gray')
axs[1].set_title("Single Centroids")

# Overlay the single centroids on the original image
axs[2].imshow(image_data, cmap='gray')
# Overlay centroids in red
centroids_y, centroids_x = zip(*centroids)
axs[2].scatter(centroids_x, centroids_y, color='red', s=50, edgecolors='black', alpha=0.75)
axs[2].set_title("Overlay of Single Centroids on Image")
plt.tight_layout()
plt.show()

print('\a')
print('\a')



#%%
mat1 = np.where(new_centroid_matrix == 0, np.nan, new_centroid_matrix)
(height, width) = mat1.shape
# Calculate the median dot-size and the median distance of two nearest dots.
(dot_size, dot_dist) = calc_size_distance(mat1, ratio=0.3)
print("Median size of dots: {0}\nMedian distance between two dots: {1}".format(
    dot_size, dot_dist))

#%%
# Reference: https://github.com/DiamondLightSource/discorpy/blob/master/examples/example_03.py
# Calculate the horizontal slope and the vertical slope of the grid.
hor_slope = prep.calc_hor_slope(mat1, ratio=0.3)
ver_slope = prep.calc_ver_slope(mat1, ratio=0.3)
print("Horizontal slope: {0}\nVertical slope: {1}".format(hor_slope, ver_slope))

#%%
print("This may take a while (Nis computer 10 minuttes)")
# Group dots into horizontal lines and vertical lines.
list_hor_lines = group_dots_hor_lines(mat1, hor_slope, dot_dist, ratio=0.4,
                                           num_dot_miss=15, accepted_ratio=0.6) # ratio=0.3, num_dot_miss=10, accepted_ratio=0.6
list_ver_lines = group_dots_ver_lines(mat1, ver_slope, dot_dist, ratio=0.4,
                                           num_dot_miss=15, accepted_ratio=0.6)

# Remove residual dots.
list_hor_lines = remove_residual_dots_hor(list_hor_lines, hor_slope,
                                               residual=2.5)
list_ver_lines = remove_residual_dots_ver(list_ver_lines, ver_slope,
                                               residual=2.5)



#%% Plot the grouping in each

# Define a list of colors to rotate through
colors_hor = itertools.cycle(['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow'])

# First figure: Original image with grouped horizontal dots
fig1 = plt.figure(figsize=(10, 6))
plt.imshow(image_data, cmap='gray')
for line in list_hor_lines:
    plt.plot(line[:, 1], line[:, 0], '-o', markersize=4, color=next(colors_hor))  # Assign next color
plt.title("Original Image with Grouped Horizontal Dots")
plt.show()  # Show the first figure

# Define a new color cycle for vertical lines
colors_ver = itertools.cycle(['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow'])

# Second figure: Original image with grouped vertical dots
fig2 = plt.figure(figsize=(10, 6))
plt.imshow(image_data, cmap='gray')
for line in list_ver_lines:
    plt.plot(line[:, 1], line[:, 0], '-o', markersize=4, color=next(colors_ver))  # Assign next color
plt.title("Original Image with Grouped Vertical Dots")
plt.show()  # Show the second figure




lengths = [len(list_hor_lines[i]) for i in range(len(list_hor_lines))]

# Plot the lengths
plt.figure(figsize=(10, 6))
plt.plot(lengths, marker='o',color='r')
plt.title('Length of Arrays in list_hor_lines')
plt.xlabel('Index i')
plt.ylabel('Length of list_hor_lines[i]')
plt.grid(True)
plt.show()



lengths = [len(list_ver_lines[i]) for i in range(len(list_ver_lines))]

# Plot the lengths
plt.plot(lengths, marker='o',color='b')
plt.title('Length of Arrays in list_hor_lines')
plt.xlabel('Index i')
plt.ylabel('Length of list_hor_lines[i]')
plt.grid(True)
plt.show()

#%%


perspective = True # Correct perspective distortion if True
# Check if the distortion is significant.
list_hor_data = calc_residual_hor(list_hor_lines, 0.0, 0.0)
losa.save_residual_plot(output_base + "/residual_hor_before_correction.png",
                      list_hor_data, height, width)
list_ver_data = calc_residual_ver(list_ver_lines, 0.0, 0.0)
losa.save_residual_plot(output_base + "/residual_ver_before_correction.png",
                      list_ver_data, height, width)
check1 = check_distortion(list_hor_data)
check2 = check_distortion(list_ver_data)
if (not check1) and (not check2):
    print("!!! Distortion is not significant !!!")

# Optional: correct perspective effect. Only available from Discorpy 1.4
if perspective is True:
    try:
        list_hor_lines, list_ver_lines = regenerate_grid_points_parabola(
            list_hor_lines, list_ver_lines, perspective=perspective)
    except AttributeError:
        raise ValueError("Perspective correction only available "
                         "from Discorpy 1.4!!!")
#%%

# Calculate the center of distortion. xcenter is the center from the left
# of the image. ycenter is the center from the top of the image.
(xcenter, ycenter) = find_cod_coarse(list_hor_lines, list_ver_lines)
# Use fine-search if there's no perspective distortion.
# (xcenter, ycenter) = find_cod_fine(list_hor_lines, list_ver_lines, xcenter, ycenter, dot_dist)
print("\nCenter of distortion:\nx-center (from the left of the image): "
      "{0}\ny-center (from the top of the image): {1}\n".format(
        xcenter, ycenter))

# Plot the image
plt.figure(figsize=(8, 6))
plt.imshow(image_data, cmap='gray')

# Overlay the center point with a red cross
plt.scatter(xcenter, ycenter, color='red', marker='x', s=100, label="Coarse Center of Distortion")

# Labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Image with Center Marked")
plt.legend()

# Show plot
plt.show()

#%%
del centroid_matrix, centroids, centroids_x, centroids_y, axs, corrected_data_2, list_ver_data, list_hor_data, new_centroid_matrix, mat1 # save for memory

num_coef = 5  # Number of polynomial coefficients
# Calculate distortion coefficients of a backward-from-forward model.
print("Calculate distortion coefficients...")
list_ffact, list_bfact = calc_coef_backward_from_forward(list_hor_lines,
                                                              list_ver_lines,
                                                              xcenter, ycenter,
                                                              num_coef,threshold=3.8)# 1.9 or 3.9

#list_fact = calc_coef_backward(list_hor_lines,list_ver_lines, xcenter, ycenter, num_coef, threshold=3.9)
#%%
# Apply distortion correction
corrected_mat = unwarp_image_backward(mat0, xcenter, ycenter, list_bfact)
#corrected_mat_bw = unwarp_image_backward(mat0, xcenter, ycenter, list_fact)

#%%
losa.save_image(output_base + "/corrected_image.tif", corrected_mat)
losa.save_image(output_base + "/diff_corrected_image.tif",
              np.abs(corrected_mat - mat0))
losa.save_metadata_txt(output_base + "/distortion_coefficients_bw.txt", xcenter,
                     ycenter, list_bfact)

# Check the correction results.
list_uhor_lines = unwarp_line_forward(list_hor_lines, xcenter, ycenter,
                                            list_ffact)
list_uver_lines = unwarp_line_forward(list_ver_lines, xcenter, ycenter,
                                            list_ffact)
losa.save_plot_image(output_base + "/horizontal_dots_unwarped.png",
                   list_uhor_lines, height, width)
losa.save_plot_image(output_base + "/vertical_dots_unwarped.png",
                   list_uver_lines, height, width)
list_hor_data = calc_residual_hor(list_uhor_lines, xcenter, ycenter)
list_ver_data = calc_residual_ver(list_uver_lines, xcenter, ycenter)
losa.save_residual_plot(output_base + "/residual_hor_after_correction.png",
                      list_hor_data, height, width)
losa.save_residual_plot(output_base + "/residual_ver_after_correction.png",
                      list_ver_data, height, width)
check1 = check_distortion(list_hor_data)
check2 = check_distortion(list_ver_data)

if check1 or check2:
    print("!!! Correction results are not at sub-pixel accuracy !!!")
time_stop = timeit.default_timer()
print("Done!!!\nRunning time is {} second!".format(time_stop - time_start))
#%%

# Plot the image

plt.figure(figsize=(8, 6))
plt.imshow(corrected_mat-mat0)
plt.show()
