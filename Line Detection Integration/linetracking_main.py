import homography as hm
import numpy as np
import touch as th
import cv2

def manual_track(dst_path, src_path):
    dst_pts = th.main(dst_path)
    src_pts = th.main(src_path)

    transformed_image = hm.transform(src_path,dst_path,np.array(src_pts[0]),np.array(dst_pts[0]))

    return transformed_image

dst_path = 'images/field2.jpeg'  #birdseye view of field
src_path  = 'images/test1.png' # frame of the video

transformed_image = manual_track(dst_path,src_path)
cv2.imshow('Transformed Birdseye View', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
