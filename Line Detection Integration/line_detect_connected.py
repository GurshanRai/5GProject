import cv2
import numpy as np
from skimage import io, color, measure

    #---------------
'''
def remove_outliers(dst_path):
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
        val = labeled_image[region.coords[0,0],region.coords[0,1]]

        bbox_ratio_comp = 1-region.area_filled/region.area_bbox #Take the compliment of the ratio of pixel_area/bbox_area 

        if( bbox_ratio_comp >= .8): # High bbox_ratio_comp suggests a lot of empty space, meaning only thin lines should appear here.
            lines.append(val)
        else:
            filtered_mask= np.where(labeled_image == val)
            labeled_image[filtered_mask[0],filtered_mask[1]] = 0 #set those outliers as background 
            
    return labeled_image
'''
def remove_outliers(gray_image):
    image_mask = gray_image > 0
    labeled_image = measure.label(image_mask,connectivity=2)

    # Find and display connected components

    lines = []
    regions = measure.regionprops(labeled_image)
    for region in regions:
        val = labeled_image[region.coords[0,0],region.coords[0,1]]

        bbox_ratio_comp = 1-region.area_filled/region.area_bbox #Take the compliment of the ratio of pixel_area/bbox_area 

        if( bbox_ratio_comp >= .8): # High bbox_ratio_comp suggests a lot of empty space, meaning only thin lines should appear here.
            lines.append(val)
        else:
            filtered_mask= np.where(labeled_image == val)
            labeled_image[filtered_mask[0],filtered_mask[1]] = 0 #set those outliers as background 
            
    return labeled_image
'''
dst_path = 'black_white.jpg' # frame of the video
labeled_image= remove_outliers(dst_path)
#print(lines)
#io.imshow(image)
# Adjust cmap as needed
io.show()
'''