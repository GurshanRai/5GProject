"""
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np

from manual_lineTracking import manual_track
from line_detect_connected import remove_outliers
import os

def main(argv):
    default_file = r'5GProject/videos/My_Movie_6.mov' # example file to replace
    filename = argv[0] if len(argv) > 0 else default_file
    # Determine if the input is an image or video
    is_video = filename.endswith(('.mp4', '.avi', '.mkv','.mov')) # for video choices

    if is_video:
        # If the input is a video, read from video file
        cap = cv.VideoCapture(filename)
        if not cap.isOpened():
            print('Error opening video file!')
            return -1 
        
        #get dimensions for the video
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))

        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for MP4


        base_name, _ = os.path.splitext(filename)
        automated_lines_path = os.path.basename(base_name+"_auto_lines") + ".mp4"
        manual_lines_path =  os.path.basename(base_name+"_manual_lines") + ".mp4"

        automated_lines_path = os.path.join('videos', automated_lines_path)
        manual_lines_path = os.path.join('videos', manual_lines_path)
        
        automated_lines = cv.VideoWriter(automated_lines_path, fourcc, fps, (frame_width, frame_height))
        manual_lines = cv.VideoWriter(manual_lines_path, fourcc, fps, (frame_width, frame_height))

    else:
        # If the input is an image, read the image
        src = cv.imread(cv.samples.findFile(filename))
        # Check if image is loaded fine
        if src is None:
            print('Error opening image!')
            print('Usage: hough_lines.py [image_name or video_name -- default ' + default_file + '] \n')
            return -1
        
    manual_track_image= None

    control = True
    while control:
        if is_video:
            # Read frame from video
            ret, src = cap.read() # video reading part
            control= ret
            if not ret:
                print('End of video')
                cap.release()
                manual_lines.release()
                automated_lines.release()
                break 
        else:
            control= False

        # Convert the frame to grayscale
        if len(src.shape) == 3:
            gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        else:
            gray = src

        if (manual_track_image is None):
            #inject manual image processing code
            manual_track_image,rect = manual_track('5GProject/images/field.jpg',src)
    
        gray =remove_outliers(src)

        # Apply Canny edge detection an opencv function
        edges = cv.Canny(gray, 50, 200, None, 3)

        # Copy edges to the images that will display the results in BGR
        cdst = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        # Standard Hough Line Transform
        # this is the line detection part
        lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

        # this is the part where the line is drawn for the first houghlines function
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        # Probabilistic Line finding
        linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

        # for the probablistic part
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        # Display the different types
        #cv.imshow("Source", src)
        #cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        #cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

        if is_video:
            manual_lines.write(manual_track_image)
            automated_lines.write(cdstP)

        # code to show manual tracking
        #combined_image = cv.addWeighted(src, .5, manual_track_image, .5, 0)
        #cv.imshow('Manual tracking', combined_image)

        # Break the loop if 'Esc' key is pressed
        # only closes with the esacpe character

        '''
        if cv.waitKey(30) == 27:
            break

    if is_video:
        # Release video capture object
        cap.release()
        manual_lines.release()
        automated_lines.release()

    cv.destroyAllWindows()
    '''
    return 0


def lineTracking(src,birdseye_field_path,manual_track_image= None):
    '''
    Parameters:
    src: image or frame of the video. Type: MatLike
    birdseye_field_path: path to a birdseye view of the field. Type: String
    manual_track_image:  Manual Line tracking. Type: uint8 NDarray (x,y,3) or None 

    Returns:
    cdstP: Automated line tracking. Type: uint8 NDarray of shape (x,y,3)
    manual_track_image: Manual Line tracking. Type: uint8 NDarray (x,y,3)

    '''

    # Convert the frame to grayscale
    if len(src.shape) == 3:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src

    if (manual_track_image is None):
        #inject manual image processing code
        manual_track_image,rect = manual_track(birdseye_field_path,src)

    gray =remove_outliers(src)

    # Apply Canny edge detection an opencv function
    edges = cv.Canny(gray, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    # Standard Hough Line Transform
    # this is the line detection part
    lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

    # this is the part where the line is drawn for the first houghlines function
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    # Probabilistic Line finding
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

    # for the probablistic part
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    return cdstP,manual_track_image


def manual_Tracking(argv):
    default_file = r'5GProject/videos/My_Movie_6.mov' # example file to replace
    filename = argv[0] if len(argv) > 0 else default_file
    # Determine if the input is an image or video
    is_video = filename.endswith(('.mp4', '.avi', '.mkv','.mov')) # for video choices

    if is_video:
        # If the input is a video, read from video file
        cap = cv.VideoCapture(filename)
        if not cap.isOpened():
            print('Error opening video file!')
            return -1 
        
        #get dimensions for the video
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))

        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for MP4


        base_name, _ = os.path.splitext(filename)
        manual_lines_path =  os.path.basename(base_name+"_manual_lines") + ".mp4"

        manual_lines_path = os.path.join('/home/bayan/Desktop/Coding_Projects/Seattle U/Team Project Capstone of Doom/5GProject/line_tracking/videos', manual_lines_path)
        
        manual_lines = cv.VideoWriter(manual_lines_path, fourcc, fps, (frame_width, frame_height))

    else:
        # If the input is an image, read the image
        src = cv.imread(cv.samples.findFile(filename))
        # Check if image is loaded fine
        if src is None:
            print('Error opening image!')
            print('Usage: hough_lines.py [image_name or video_name -- default ' + default_file + '] \n')
            return -1
        
    manual_track_image= None

    control = True
    while control:
        if is_video:
            # Read frame from video
            ret, src = cap.read() # video reading part
            control= ret
            if not ret:
                print('End of video')
                cap.release()
                manual_lines.release()
                break 
        else:
            control= False

        if (manual_track_image is None):
            #inject manual image processing code
            manual_track_image,rect = manual_track('5GProject/images/field.jpg',src)
    

        # code to show manual tracking
        combined_image = cv.addWeighted(src, .5, manual_track_image, .5, 0)

        if is_video:
            manual_lines.write(combined_image)


        #cv.imshow('Manual tracking', combined_image)

        # Break the loop if 'Esc' key is pressed
        # only closes with the esacpe character

        '''
        if cv.waitKey(30) == 27:
            break

    if is_video:
        # Release video capture object
        cap.release()
        manual_lines.release()
        automated_lines.release()

    cv.destroyAllWindows()
    '''
    return 0

if __name__ == "__main__":
    #main(sys.argv[1:])
    manual_Tracking(sys.argv[1:])
