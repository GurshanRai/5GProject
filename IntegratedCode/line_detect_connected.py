import cv2
import numpy as np
from skimage import io, color, measure
from image_processing import process_image

    #---------------
def remove_outliers(image):
    '''
    if(isinstance(image,str)):
        image = io.imread(image)
    '''

    processed_image = process_image(image)

    if len(processed_image.shape) ==3:
        gray_image = color.rgb2gray(processed_image)
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

        if( bbox_ratio_comp >= .80): # High bbox_ratio_comp suggests a lot of empty space, meaning only thin lines should appear here.
            lines.append(val)
        else:
            filtered_mask= np.where(labeled_image == val)
            labeled_image[filtered_mask[0],filtered_mask[1]] = 0 #set those outliers as background 
            gray_image[filtered_mask[0],filtered_mask[1]] *= 0
        
        #io.imshow(gray_image)
        #io.imshow(labeled_image, cmap='nipy_spectral', alpha=1)  # Adjust cmap as needed
        #io.show()

    gray_image = (gray_image * 255).astype(np.uint8)
    filtered_mask= np.where(gray_image < 200 )
    gray_image[filtered_mask[0],filtered_mask[1]] =0
    return gray_image


'''
dst_path = 'black_white.jpg' # frame of the video

labeled_image= remove_outliers(dst_path)
#print(lines)
io.imshow(io.imread(dst_path))
io.imshow(labeled_image, cmap='nipy_spectral', alpha=1)
# Adjust cmap as needed
io.show()
'''