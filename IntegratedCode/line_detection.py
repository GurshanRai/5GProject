"""
@brief This program demonstrates line finding with the Hough and Homography transforms
"""
import sys
import math
import cv2 as cv
import numpy as np

from linetracking_main import manual_track
from line_detect_connected import remove_outliers
import track_ball
import os

#the main function runs both the manual and automated tracking
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

        #create names for the processed videos
        base_name, _ = os.path.splitext(filename)
        automated_lines_path = os.path.basename(base_name+"_auto_lines") + ".mp4" #automated tracking gets tagged with _auto_lines
        manual_lines_path =  os.path.basename(base_name+"_manual_lines") + ".mp4" #manual tracking gets tagged with _manual_lines

        automated_lines_path = os.path.join('videos', automated_lines_path)
        manual_lines_path = os.path.join('videos', manual_lines_path)
        
        #video is saved in an mp4 format in the same directory as the source video
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
            manual_track_image,rect = manual_track('field.jpg',src)
    
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

        if is_video:

            #write the result of the manual tracking to the repective video files
            manual_lines.write(manual_track_image)
            automated_lines.write(cdstP)

    return 0


#operates in the same way as main, but the resulting images are exportable directly using a function call
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


#only performs the manual tracking on a given video
def manual_Tracking(path,output_videos_path):
    '''
    Brief:
    Takes in a video path. Performs manual tracking using a Homography transform to warp the map of the field. The output video has the clairified lines on the field. 
    The location of the output video is specified by output_videos_path.

    Parameters:
    path: path to the raw video/stream .
    output_videos_path: output directory of the manually tracked videos.

    Returns:
    errorCode: returns 0 if successful, returns -1 if an error occured.

    '''
    filename = path
    # Determine if the input is an image or video
    is_video = filename.endswith(('.mp4', '.avi', '.mkv','.mov','.MOV')) # for video choices

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

        base_name, ext = os.path.splitext(filename)
        manual_lines_path =  os.path.basename(base_name) + ext
 
        manual_lines_path = os.path.join(output_videos_path, manual_lines_path)
        
        manual_lines = cv.VideoWriter(manual_lines_path, fourcc, fps, (frame_width, frame_height))

    else:
        # If the input is an image, read the image
        src = cv.imread(cv.samples.findFile(filename))
        # Check if image is loaded fine
        if src is None:
            print('Error opening image!')
            print('Usage: hough_lines.py [image_name or video_name -- default ' + "filename" + '] \n')
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
            manual_track_image,rect = manual_track('field.jpg',src)
    

        # code to show manual tracking
        combined_image =  cv.bitwise_or(manual_track_image,src)

        if is_video:
            manual_lines.write(combined_image)

    return 0


'''
The user runs the program like so:
python3 line_detection <videos folder> <line detection output directory> <ball tracking output directory>.
'''
if __name__ == "__main__":
    videos = os.listdir(sys.argv[1]) #take in videos folder as the 1st argument
    paths = []
    video_names = []
    
    for video in videos:
        
        video_names.append(video)
        path = sys.argv[2] + "/" + video #take in the output folder as the 2nd argument
        paths.append(path)
        manual_Tracking(sys.argv[1] + "/" + video, sys.argv[2]) # videos will be saved as <line detection folder>/videoname. The video name remains the same.

    print(paths)
    output_folder = sys.argv[3] #take in output directory for ball tracking
    track_ball.process_videos([paths,output_folder]) # after line tracking is concluded, the output is pipelined into the ball tracking algorithm
    track_ball.save_output(output_folder,video_names) # output of the videos are saved.
