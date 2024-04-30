import homography as hm
import numpy as np
import touch as th
import cv2

def manual_track(src_path,dst_path):
    '''
    Parameters:
    src_path: string path to source image (Image that's transformed) or a Matlike of the image
    dst_path: string path to destination image (Image that source is being warped onto) or a Matlike of the image
    
    Return:
    transformed_image: Matlike representing the transformed image
    rect_drawer: Rectangle drawing class, used to draw lines on an image
    '''
    dst_pts,_ = th.main(dst_path)
    src_pts, rect_drawer = th.main(src_path)

    transformed_image = hm.transform(src_path,dst_path,np.array(src_pts[0]),np.array(dst_pts[0]))

    return transformed_image,rect_drawer

'''
dst_path = 'images/field2.jpeg'  #birdseye view of field
src_path  = 'images/test1.png' # frame of the video

transformed_image,_ = manual_track(dst_path,src_path)
cv2.imshow('Transformed Birdseye View', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''