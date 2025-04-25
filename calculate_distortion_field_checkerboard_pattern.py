import os
# Change to your local directory:
os.chdir(r"C:\Users\nige\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Postdoc\Projects\Detector paper\20241204 beamtimedata\process\fields_corrected\MLL_new_detector")

# Import the distortion functions:
from distortion_functions import *

# Load checkerboard pattern:
os.chdir(r"C:\Users\nige\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Postdoc\Projects\Detector paper\20250225 beamtimedata\process\field_corrected")
fnames = ['grid_scan_2398_corrected.h5'] # Martins Bech Checkerboard patter, measured at DanMAX February 2025

with h5py.File(fnames[0], 'r') as file:
    image_data = file['corrected/image']
    corrected_data_2 = image_data[:]

# Some images and the distortion field are saved. Select where you want them to be saved: 
output_base = "../"

#%% Plot Checkerboard pattern and zoom-in
fs=14    
# Create the subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot the full data
im = axs[0].imshow(corrected_data_2, cmap='gray')
axs[0].set_title("Original image",fontsize=fs)
axs[0].set_xlabel("X-axis (pixels)",fontsize=fs)
axs[0].set_ylabel("Y-axis (pixels)",fontsize=fs)

# Plot a cropped section of the data
axs[1].imshow(corrected_data_2[3000:3400, 3000:3400], cmap='gray')
axs[1].set_title("Zoom-in Section of Original Image",fontsize=fs)
axs[1].set_xlabel("X-axis (pixels)",fontsize=fs)
axs[1].set_ylabel("Y-axis (pixels)",fontsize=fs)

plt.tight_layout() 
plt.show()


#%%

# Select data ROI
image_data = corrected_data_2[0:-1,0:-1] # (10640, 13856)  # cod: (6787,6921)
image_data = np.where(np.isnan(image_data), 0.4, image_data) # Remove empty pixels
mat0 = image_data # Initiate to use Discorypy nameing        https://github.com/DiamondLightSource/discorpy/tree/master

centroid_matrix = harris_corners_to_centroid_matrix2(image_data, square_size=23, corner_threshold_factor=0.001) # Find corners using CV2 HarrisCorners. 
centroids = compute_centroids_from_matrix2(centroid_matrix) # Find Centroids. 

# Step 3: Create a new matrix with a single dot for each centroid
new_centroid_matrix = np.zeros_like(centroid_matrix)
for (cy, cx) in centroids:
    new_centroid_matrix[int(cy), int(cx)] = 1  # Mark the centroid in the new matrix

# Visualization with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
# Plot the original image
axs[0, 0].imshow(image_data, cmap='gray')
axs[0, 0].set_title("Original Image", fontsize=fs)
axs[0, 0].set_xlabel("X-axis (pixels)", fontsize=fs)
axs[0, 0].set_ylabel("Y-axis (pixels)", fontsize=fs)
# Plot the new centroid matrix with single centroids
axs[0, 1].imshow(new_centroid_matrix, cmap='gray')
axs[0, 1].set_title("Single Centroids", fontsize=fs)
axs[0, 1].set_xlabel("X-axis (pixels)", fontsize=fs)
axs[0, 1].set_ylabel("Y-axis (pixels)", fontsize=fs)
# Overlay the single centroids on the original image
axs[1, 0].imshow(image_data, cmap='gray')
# Overlay centroids in red
centroids_y, centroids_x = zip(*centroids)
axs[1, 0].scatter(centroids_x, centroids_y, color='red', s=50, edgecolors='black', alpha=0.75)
axs[1, 0].set_title("Overlay of Single Centroids on Original Image", fontsize=fs)
axs[1, 0].set_xlabel("X-axis (pixels)", fontsize=fs)
axs[1, 0].set_ylabel("Y-axis (pixels)", fontsize=fs)

# Plot the zoomed-in area [3000:3400, 3000:3400] as the fourth subplot (axs[1, 1])
zoomed_in_area = image_data[3000:3400, 3000:3400]  # Zoom into the specified area
axs[1, 1].imshow(zoomed_in_area, cmap='gray')  # Display zoomed-in image
# Overlay centroids in red (zoomed-in area)
centroids_y_zoomed, centroids_x_zoomed = zip(*[(y - 3000, x - 3000) for y, x in centroids if 3000 <= y <= 3400 and 3000 <= x <= 3400])
axs[1, 1].scatter(centroids_x_zoomed, centroids_y_zoomed, color='red', s=50, edgecolors='black', alpha=0.75)
axs[1, 1].set_title("Zoomed-In Area with Centroids", fontsize=fs)
axs[1, 1].set_xlabel("X-axis (pixels)", fontsize=fs)
axs[1, 1].set_ylabel("Y-axis (pixels)", fontsize=fs)
plt.tight_layout()
plt.show()

#%%

mat1 = np.nan_to_num(new_centroid_matrix, nan=0, posinf=0, neginf=0)
(height, width) = mat1.shape
# Calculate the median dot-size and the median distance of two nearest dots.
(dot_size, dot_dist) = calc_size_distance(mat1, ratio=0.3)
print("Median size of dots: {0}\nMedian distance between two dots: {1}".format(
    dot_size, dot_dist))

#%%
# Reference: https://github.com/DiamondLightSource/discorpy/blob/master/examples/example_03.py
# Calculate the horizontal slope and the vertical slope of the grid.
hor_slope = prep.calc_hor_slope(mat1, ratio=0.3) # ratio=0.3
ver_slope = prep.calc_ver_slope(mat1, ratio=0.3) # ratio=0.3
print("Horizontal slope: {0}\nVertical slope: {1}".format(hor_slope, ver_slope))

#%%
print("This may take a while (Nis computer 8 minuttes)")
# Group dots into horizontal lines and vertical lines.
list_hor_lines = group_dots_hor_lines(mat1, hor_slope, dot_dist, ratio=0.1,
                                           num_dot_miss=15, accepted_ratio=0.6) # ratio=0.3, num_dot_miss=10, accepted_ratio=0.6
list_ver_lines = group_dots_ver_lines(mat1, ver_slope, dot_dist, ratio=0.1,
                                           num_dot_miss=15, accepted_ratio=0.6)

# Remove residual dots.
list_hor_lines = remove_residual_dots_hor(list_hor_lines, hor_slope, residual=2.5)
list_ver_lines = remove_residual_dots_ver(list_ver_lines, ver_slope, residual=2.5)


# Plot the grouping in each
# Define a list of colors to rotate through
colors_hor = itertools.cycle(['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow'])

# First figure: Original image with grouped horizontal dots
fig1 = plt.figure(figsize=(10, 6))
plt.imshow(image_data, cmap='gray')
for line in list_hor_lines:
    plt.plot(line[:, 1], line[:, 0], '-o', markersize=4, color=next(colors_hor))  # Assign next color
plt.title("Original Image with Grouped Horizontal Dots", fontsize=fs)
plt.xlabel("X-axis (pixels)", fontsize=fs)
plt.ylabel("Y-axis (pixels)", fontsize=fs)
plt.show()  # Show the first figure

# Define a new color cycle for vertical lines
colors_ver = itertools.cycle(['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow'])

# Second figure: Original image with grouped vertical dots
fig2 = plt.figure(figsize=(10, 6))
plt.imshow(image_data, cmap='gray')
for line in list_ver_lines:
    plt.plot(line[:, 1], line[:, 0], '-o', markersize=4, color=next(colors_ver))  # Assign next color
plt.title("Original Image with Grouped Vertical Dots", fontsize=fs)
plt.xlabel("X-axis (pixels)", fontsize=fs)
plt.ylabel("Y-axis (pixels)", fontsize=fs)
plt.show()  # Show the second figure

lengths_hor = [len(list_hor_lines[i]) for i in range(len(list_hor_lines))]
lengths_ver = [len(list_ver_lines[i]) for i in range(len(list_ver_lines))]
# Plot the lengths
plt.figure(figsize=(10, 6))
plt.plot(lengths_hor, marker='o', color='r', label='Horizontal Lines')
plt.plot(lengths_ver, marker='o', color='b', label='Vertical Lines')
# Title and axis labels
plt.title('Length of Arrays in Grouped Vertical and Horizontal Dots', fontsize=fs)
plt.xlabel('Index i', fontsize=fs)
plt.ylabel('Length of Arrays', fontsize=fs)
# Add a grid
plt.grid(True)
# Add legend
plt.legend()
# Show the plot
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
print("Checking horizontal")
check1 = check_distortion(list_hor_data)
print("Checking vertical")
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
        
# Plotting the residuals for horizontal lines
plt.figure(figsize=(10, 6))
plt.plot(list_hor_data[:, 0], list_hor_data[:, 1], 'r.', label='Horizontal Residuals', markersize=4)
plt.title('Residuals of Horizontal Lines', fontsize=14)
plt.xlabel('Radius (pixels)', fontsize=12)
plt.ylabel('Residual (pixels)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the residuals for vertical lines
plt.figure(figsize=(10, 6))
plt.plot(list_ver_data[:, 0], list_ver_data[:, 1], 'b.', label='Vertical Residuals', markersize=4)
plt.title('Residuals of Vertical Lines', fontsize=14)
plt.xlabel('Radius (pixels)', fontsize=12)
plt.ylabel('Residual (pixels)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



#%%

# Calculate the center of distortion. xcenter is the center from the left
# of the image. ycenter is the center from the top of the image.
(xcenter, ycenter) = find_cod_coarse(list_hor_lines, list_ver_lines)
print("\nCenter of distortion (Coarse):\nx-center (from the left of the image): "
      "{0}\ny-center (from the top of the image): {1}\n".format(
        xcenter, ycenter))

# Use bailey-search to find CoD.
(xcenter, ycenter) = find_cod_bailey(list_hor_lines, list_ver_lines, iteration=100)
print("\nCenter of distortion (Bailey):\nx-center (from the left of the image): "
      "{0}\ny-center (from the top of the image): {1}\n".format(
        xcenter, ycenter))

# Use fine-search if there's no perspective distortion.
(xcenter, ycenter) = find_cod_fine(list_hor_lines, list_ver_lines, xcenter, ycenter, dot_dist)
print("\nCenter of distortion (Fine):\nx-center (from the left of the image): "
      "{0}\ny-center (from the top of the image): {1}\n".format(
        xcenter, ycenter))

# Plot the image
plt.figure(figsize=(8, 6))
plt.imshow(image_data, cmap='gray')
# Overlay the center point with a red cross
plt.scatter(xcenter, ycenter, color='red', marker='x', s=100, label="Coarse Center of Distortion")
plt.xlabel("X-axis",fontsize=fs)
plt.ylabel("Y-axis",fontsize=fs)
plt.title("Original Image with Center Marked",fontsize=fs)
plt.legend()
plt.show()


del centroids, centroids_x, centroids_y, axs, corrected_data_2, list_ver_data, list_hor_data, new_centroid_matrix, mat1, centroid_matrix, centroids_x_zoomed, centroids_y_zoomed, lengths_ver  # Nis had to do this to save memory on his computer 
num_coef = 5  # Number of polynomial coefficients
# Calculate distortion coefficients of a backward-from-forward model.
print("Calculate distortion coefficients...")
list_ffact, list_bfact = calc_coef_backward_from_forward(list_hor_lines,
                                                              list_ver_lines,
                                                              xcenter, ycenter,
                                                              num_coef,threshold=5.8)# 1.9 or 3.9

# Apply field distortion correction
indices = compute_indices(xcenter, ycenter, list_bfact, mat0.shape) # New function. Get distortion indicies
corrected_mat = unwarp_image_backward_tester(mat0, indices) # New function. Apply distortion field. This should be called on each projection

# Save distortion field coefficients
losa.save_metadata_txt(output_base + "/distortion_coefficients_bw.txt", xcenter, ycenter, list_bfact)

# Check the correction results. #unwarp_line_backward or unwarp_line_forward
list_uhor_lines = unwarp_line_backward(list_hor_lines, xcenter, ycenter,
                                            list_bfact)
list_uver_lines = unwarp_line_backward(list_ver_lines, xcenter, ycenter,
                                            list_bfact)

list_hor_data = calc_residual_hor(list_uhor_lines, xcenter, ycenter)
list_ver_data = calc_residual_ver(list_uver_lines, xcenter, ycenter)

print("Checking horizontal")
check1 = check_distortion(list_hor_data) # This line checks if the percentage of residuals greater than 1.0 exceeds 15%
print("Checking vertical")
check2 = check_distortion(list_ver_data) # This line checks if the percentage of residuals greater than 1.0 exceeds 15%

if check1 or check2:
    print("!!! OBS: Correction results are not at sub-pixel accuracy !!!")
else: 
    print("Correction results are at sub-pixel accuracy.")


#%%

# Plot the corrected image
plt.figure(figsize=(8, 6))
plt.imshow(corrected_mat, cmap='gray')
plt.title('Corrected Image', fontsize=14)
plt.xlabel('X-axis (pixels)', fontsize=12)
plt.ylabel('Y-axis (pixels)', fontsize=12)
plt.colorbar(label='Intensity')  # Optional: Add colorbar for intensity scale
plt.tight_layout()
plt.show()

# Plot the difference between the corrected and original image
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(corrected_mat - mat0), cmap='hot')  # Hot colormap for better visibility
plt.title('Difference Between Corrected and Original Image', fontsize=14)
plt.xlabel('X-axis (pixels)', fontsize=12)
plt.ylabel('Y-axis (pixels)', fontsize=12)
plt.colorbar(label='Intensity Difference')  # Optional: Add colorbar for difference scale
plt.tight_layout()
plt.show()

# Plotting the residuals for horizontal lines
plt.figure(figsize=(10, 6))
plt.plot(list_hor_data[:, 0], list_hor_data[:, 1], 'r.', label='Horizontal Residuals', markersize=4)
plt.title('Residuals of Horizontal Lines after correction', fontsize=14)
plt.xlabel('Radius (pixels)', fontsize=12)
plt.ylabel('Residual (pixels)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the residuals for vertical lines
plt.figure(figsize=(10, 6))
plt.plot(list_ver_data[:, 0], list_ver_data[:, 1], 'b.', label='Vertical Residuals', markersize=4)
plt.title('Residuals of Vertical Lines after correction', fontsize=14)
plt.xlabel('Radius (pixels)', fontsize=12)
plt.ylabel('Residual (pixels)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Plot for the horizontal lines (nearest to ycenter, top, and bottom)
fig1 = plt.figure(figsize=(10, 6))

# Find the horizontal line closest to ycenter
nearest_line = min(list_hor_lines, key=lambda line: abs(np.mean(line[:, 0]) - ycenter))

# Normalize the y-values of the nearest horizontal line
nearest_line_normalized = nearest_line.copy()
nearest_line_normalized[:, 0] -= nearest_line[0, 0]  # Subtract the first y value to normalize

# Plot the nearest line (near ycenter) in blue
plt.plot(nearest_line_normalized[:, 1], nearest_line_normalized[:, 0], '-o', markersize=4, color='blue', label='Nearest to ycenter')

# Normalize the y-values of the top horizontal line
top_line = list_hor_lines[-30]
top_line_normalized = top_line.copy()
top_line_normalized[:, 0] -= top_line[0, 0]  # Subtract the first y value to normalize
plt.plot(top_line_normalized[:, 1], top_line_normalized[:, 0], '-o', markersize=4, color='red', label='Line near the Top')

# Normalize the y-values of the bottom horizontal line
bottom_line = list_hor_lines[30]
bottom_line_normalized = bottom_line.copy()
bottom_line_normalized[:, 0] -= bottom_line[0, 0]  # Subtract the first y value to normalize
plt.plot(bottom_line_normalized[:, 1], bottom_line_normalized[:, 0], '-o', markersize=4, color='green', label='Line near the Bottom')

# Adjust the axis to match the image orientation
plt.gca().invert_yaxis()  # Invert the y-axis to match the image coordinates (top to bottom)
plt.gca().set_aspect('auto', adjustable='box')  # Adjust aspect ratio if needed

# Adding title and labels
plt.title("Grouped Horizontal Dots (Nearest to ycenter, Top and Bottom) Normalized to the ycenter", fontsize=14)
plt.xlabel("X-axis (pixels)", fontsize=12)
plt.ylabel("Y-axis (pixels)", fontsize=12)

# Rotate the y-axis ticks by 90 degrees
plt.tick_params(axis='y', rotation=90)
plt.legend(fontsize=fs)
plt.show()

# Plot for the vertical lines (nearest to xcenter, left, and right)
fig2 = plt.figure(figsize=(10, 6))

# Find the vertical line closest to xcenter
nearest_ver_line = min(list_ver_lines, key=lambda line: abs(np.mean(line[:, 1]) - xcenter))

# Normalize the x-values of the nearest vertical line
nearest_ver_line_normalized = nearest_ver_line.copy()
nearest_ver_line_normalized[:, 1] -= nearest_ver_line[0, 1]  # Subtract the first x value to normalize

# Plot the nearest vertical line (near xcenter) in cyan
plt.plot(nearest_ver_line_normalized[:, 1], nearest_ver_line_normalized[:, 0], '-o', markersize=4, color='blue', label='Nearest to xcenter')

# Normalize the x-values of the left vertical line
top_ver_line = list_ver_lines[-30]
top_ver_line_normalized = top_ver_line.copy()
top_ver_line_normalized[:, 1] -= top_ver_line[0, 1]  # Subtract the first x value to normalize
plt.plot(top_ver_line_normalized[:, 1], top_ver_line_normalized[:, 0], '-o', markersize=4, color='red', label='Line near the Left')

# Normalize the x-values of the right vertical line
bottom_ver_line = list_ver_lines[30]
bottom_ver_line_normalized = bottom_ver_line.copy()
bottom_ver_line_normalized[:, 1] -= bottom_ver_line[0, 1]  # Subtract the first x value to normalize
plt.plot(bottom_ver_line_normalized[:, 1], bottom_ver_line_normalized[:, 0], '-o', markersize=4, color='green', label='Line near the Right')

# Invert the y-axis to match the image coordinates (top to bottom)
plt.gca().invert_yaxis()

# Rotate the y-axis ticks by 90 degrees
plt.tick_params(axis='y', rotation=90)

# Adding title and labels
plt.title("Grouped Vertical Dots (Nearest to xcenter, Left and Right) Normalized to the xcenter", fontsize=14)
plt.xlabel("X-axis (pixels)", fontsize=12)
plt.ylabel("Y-axis (pixels)", fontsize=12)
plt.legend(fontsize=fs)
plt.show()


# Plot for the corrected horizontal lines (nearest to ycenter, top, and bottom)
fig1 = plt.figure(figsize=(10, 6))

# Find the corrected horizontal line closest to ycenter
nearest_line = min(list_uhor_lines, key=lambda line: abs(np.mean(line[:, 0]) - ycenter))

# Normalize the y-values of the nearest corrected horizontal line
nearest_line_normalized = nearest_line.copy()
nearest_line_normalized[:, 0] -= nearest_line[0, 0]  # Subtract the first y value to normalize

# Plot the nearest corrected line (near ycenter) in blue
plt.plot(nearest_line_normalized[:, 1], nearest_line_normalized[:, 0], '-o', markersize=4, color='blue', label='Nearest to ycenter')

# Normalize the y-values of the top corrected line
top_line = list_uhor_lines[-30]
top_line_normalized = top_line.copy()
top_line_normalized[:, 0] -= top_line[0, 0]  # Subtract the first y value to normalize
plt.plot(top_line_normalized[:, 1], top_line_normalized[:, 0], '-o', markersize=4, color='red', label='Line near the Top')

# Normalize the y-values of the bottom corrected line
bottom_line = list_uhor_lines[30]
bottom_line_normalized = bottom_line.copy()
bottom_line_normalized[:, 0] -= bottom_line[0, 0]  # Subtract the first y value to normalize
plt.plot(bottom_line_normalized[:, 1], bottom_line_normalized[:, 0], '-o', markersize=4, color='green', label='Line near the Bottom')

# Adjust the axis to match the image orientation
plt.gca().invert_yaxis()  # Invert the y-axis to match the image coordinates (top to bottom)
plt.gca().set_aspect('auto', adjustable='box')  # Adjust aspect ratio if needed
plt.title("Corrected Horizontal Dots (Nearest to ycenter, Top and Bottom) Normalized to the ycenter", fontsize=14)
plt.xlabel("X-axis (pixels)", fontsize=12)
plt.ylabel("Y-axis (pixels)", fontsize=12)

# Rotate the y-axis ticks by 90 degrees
plt.tick_params(axis='y', rotation=90)
plt.legend(fontsize=fs)
plt.show()

# Plot for the corrected vertical lines (nearest to xcenter, left, and right)
fig2 = plt.figure(figsize=(10, 6))
# Find the corrected vertical line closest to xcenter
nearest_ver_line = min(list_uver_lines, key=lambda line: abs(np.mean(line[:, 1]) - xcenter))
# Normalize the x-values of the nearest corrected vertical line
nearest_ver_line_normalized = nearest_ver_line.copy()
nearest_ver_line_normalized[:, 1] -= nearest_ver_line[0, 1]  # Subtract the first x value to normalize

# Plot the nearest corrected vertical line (near xcenter) in cyan
plt.plot(nearest_ver_line_normalized[:, 1], nearest_ver_line_normalized[:, 0], '-o', markersize=4, color='blue', label='Nearest to xcenter')

# Normalize the x-values of the left corrected vertical line
top_ver_line = list_uver_lines[-30]
top_ver_line_normalized = top_ver_line.copy()
top_ver_line_normalized[:, 1] -= top_ver_line[0, 1]  # Subtract the first x value to normalize
plt.plot(top_ver_line_normalized[:, 1], top_ver_line_normalized[:, 0], '-o', markersize=4, color='red', label='Line near the Left')

# Normalize the x-values of the right corrected vertical line
bottom_ver_line = list_uver_lines[30]
bottom_ver_line_normalized = bottom_ver_line.copy()
bottom_ver_line_normalized[:, 1] -= bottom_ver_line[0, 1]  # Subtract the first x value to normalize
plt.plot(bottom_ver_line_normalized[:, 1], bottom_ver_line_normalized[:, 0], '-o', markersize=4, color='green', label='Line near the Right')

# Invert the y-axis to match the image coordinates (top to bottom)
plt.gca().invert_yaxis()

# Rotate the y-axis ticks by 90 degrees
plt.tick_params(axis='y', rotation=90)
plt.title("Corrected Vertical Dots (Nearest to xcenter, Left and Right) Normalized to the xcenter", fontsize=14)
plt.xlabel("X-axis (pixels)", fontsize=12)
plt.ylabel("Y-axis (pixels)", fontsize=12)
plt.legend(fontsize=fs)
plt.show()

#%% Not necessary to look at:


xu_list = np.arange(width) - xcenter
yu_list = np.arange(height) - ycenter
xu_mat, yu_mat = np.meshgrid(xu_list, yu_list)
ru_mat = np.sqrt(xu_mat ** 2 + yu_mat ** 2)
fact_mat = np.sum(np.asarray(
    [factor * ru_mat ** i for i, factor in enumerate(list_bfact)]), axis=0)
xd_mat = np.float32(np.clip(xcenter + fact_mat * xu_mat, 0, width - 1))
yd_mat = np.float32(np.clip(ycenter + fact_mat * yu_mat, 0, height - 1))
#%%

dx = xd_mat - xu_mat  # Horizontal displacement
dy = yd_mat - yu_mat  # Vertical displacement

# Create a quiver plot to show the distortion field
fig, ax = plt.subplots(figsize=(10, 6))

# Use quiver to plot the displacement vectors (dx, dy)
ax.quiver(xu_mat, yu_mat, dx, dy, angles='xy', scale_units='xy', scale=0.1, color='blue')

# Set plot title and labels
ax.set_title("Distortion Field as Vector Field", fontsize=14)
ax.set_xlabel("X-axis", fontsize=12)
ax.set_ylabel("Y-axis", fontsize=12)

# Invert the y-axis to match image coordinates
ax.invert_yaxis()

# Display the plot
plt.tight_layout()
plt.show()
#%%
# Calculate the magnitude of the distortion displacement (Euclidean distance)
magnitude = np.sqrt(dx**2 + dy**2)

# Create a heatmap of the magnitude of the distortion
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.imshow(magnitude, cmap='hot', interpolation='none')

# Set the title and labels
ax.set_title("Magnitude of Distortion Field", fontsize=14)
ax.set_xlabel("X-axis", fontsize=12)
ax.set_ylabel("Y-axis", fontsize=12)

# Add a colorbar to show the intensity scale for the magnitude
fig.colorbar(cax, ax=ax, label='Magnitude of Distortion')

# Invert the y-axis to match image coordinates
ax.invert_yaxis()

# Show the plot
plt.tight_layout()
plt.show()

#%%

# Assuming 'coordinates' contains the distorted coordinates (2, H, W)
# Extract the x and y components
coordinates = np.copy(indices)
# Reshape coordinates to match the mat0 shape (10639, 13855)
x_distorted = coordinates[0].reshape(mat0.shape)  # Distorted x-coordinates
y_distorted = coordinates[1].reshape(mat0.shape)  # Distorted y-coordinates

# Create a meshgrid for the original coordinates of mat0
y, x = np.meshgrid(np.arange(0, mat0.shape[0]), np.arange(0, mat0.shape[1]), indexing='ij')

# Compute the displacement in both x and y directions
displacement_x = x_distorted - x  # Difference in the x-direction
displacement_y = y_distorted - y  # Difference in the y-direction

# Plot the distortion field using a quiver plot
plt.figure(figsize=(10, 8))
plt.quiver(x, y, displacement_x, displacement_y, scale=5, pivot='middle', color='blue', headwidth=2)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Distortion Field Applied to mat0 Image")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

