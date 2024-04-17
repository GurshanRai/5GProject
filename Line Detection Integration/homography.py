import cv2
import numpy as np

def stack_pts(x_coords,y_coords):
    return np.column_stack((x_coords,y_coords))

def transform(src_path,dst_path,src_pts,dst_pts):
    '''
    Parameters:
    src_path: string path to source image (Image that's transformed)
    dst_path: string path to destination image (Image that source is being warped onto)
    src_pts: np array of points of source image
    dst_pts: np array of points from destination image, each point in dst corresponds to points in src.

    Return:
    transformed_image: Matlike representing the transformed image
    '''

    src= cv2.imread(src_path)
    dst= cv2.imread(dst_path) 

    homography_matrix, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC, 1)
    transformed_image = cv2.warpPerspective(src, homography_matrix, (dst.shape[1], dst.shape[0]))
   
    return transformed_image
