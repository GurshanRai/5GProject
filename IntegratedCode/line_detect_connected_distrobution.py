import cv2
import numpy as np
from skimage import io, color, measure

#---------------
dst_path = 'black_white.jpg' # frame of the video
image = io.imread(dst_path)

# Label connected components
if len(image.shape) ==3:
    gray_image = color.rgb2gray(image)
else:
    gray_image = image
image_mask = gray_image > .5
labeled_image = measure.label(image_mask,connectivity=2)

# Find and display connected components

lines = []
regions = measure.regionprops(labeled_image)
for region in regions:
    val = labeled_image[region.coords[0,0],region.coords[0,1]] #take value of first x,y coords, rest of the cluster is the same value

    filtered_mask= np.where(labeled_image == val) #filter only this cluster

    bbox_ratio_comp = 1-region.area_filled/region.area_bbox #Take the compliment of the ratio of pixel_area/bbox_area 
    labeled_image[filtered_mask[0],filtered_mask[1]] = int(bbox_ratio_comp*100) #relabel according to the ratio
        
print(lines)
io.imshow(image)
io.imshow(labeled_image, cmap='nipy_spectral', alpha=1)  # Adjust cmap as needed
io.show()