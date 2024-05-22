# Import the necessary modules.
# This program only requires "pip install mediapipe" to run

# WARNING: 
# VIDEOS MUST BE RELATIVELY SAME LENGTH AS THE PROGRAM WILL STOP AFTER THE SHORTEST VIDEO ENDS,
# VIDEO NAMES MUST CONTAIN NO WHITESPACE
import sys
import time
import os
import cv2
import shutil

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from screeninfo import get_monitors

# Constant variables
# Variables for key presses
NO_KEY = -1
ESC_KEY = 27
SPACE_KEY = 32
# Variables for drawing text/boxes
COLOR_MAGENTA = (255,0,255)
LINE_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SIZE = 2
TOP_LEFT = (15,60)

# Variable for path to model
MODEL_PATH = "custom_model/model/model.tflite"
# Adjust minimum confidence score as desired
SCORE_THRESHOLD = 0.55

# Create an ObjectDetector object from model
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=SCORE_THRESHOLD)

# Initialize detector from ObjectDetector object and create detector for each video
detectors = []
for _ in range(4):
    detector = vision.ObjectDetector.create_from_options(options)
    detectors.append(detector)

# Initialize tracker (can be any of the trackers offered in OpenCV) for each video
trackers = []
for _ in range(4):
    tracker = cv2.TrackerCSRT_create()
    trackers.append(tracker)

# Reads frame and camera index
# Returns the bounding box of detected ball
def detect_ball(frame, index):
    # Convert frame to RGB format in order to be processed and detected
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detectors[index].detect(rgb_frame)

    # Draw box around detected ball if any are detected
    if len(detection_result.detections) != 0:
        return detection_result.detections[0].bounding_box
    
    return None
    
# Reads frame and camera index
# Returns true if tracker succeeds, false otherwise
def track_ball(frame,index):
    # Track movement within ROI per frame
    (success,box) = trackers[index].update(frame)
    # If there is movement, update tracking box to returned box
    if success:
        (x,y,width,height) = [int(var) for var in box]
        cv2.rectangle(frame,(x,y),(x+width,y+height), COLOR_MAGENTA, LINE_THICKNESS)

        return True
    return False

def save_output(out, video_names):
    for video_name in video_names:
        frame_dir = "_temp/" + video_name + "/frame%d.png"
        os.system("ffmpeg -y -framerate 60 -i '" + frame_dir + "' -c:v libx264 -r 60 '" + out + "/" + video_name + "'")
        shutil.rmtree("_temp/" + video_name)
    os.rmdir("_temp")


def process_videos(argv):
    frame_count = 0
    videos = argv[0] 

    captures = []
    video_names = []

    for video in videos:
        video_names.append(video.split("/")[-1])
        cv2.namedWindow(video,cv2.WINDOW_NORMAL)
        # Load the input image/video into capture object
        cap = cv2.VideoCapture(video)
        captures.append(cap)

    if not cap.isOpened():
        print("Camera could not be opened\n")

    # Variables for reading each video
    ret = [None] * len(videos)
    frames = [None] * len(videos)
    ball_Detected = [False] * len(videos)
    Tracking = [False] * len(videos)
    Mode = [""] * len(videos)
    frame_count = [0] * len(videos)
    screen_corners = [(0,0)] * len(videos)
    
    # Get screen resolution for window placement
    monitor = get_monitors()
    screen_corners[0] = (monitor[0].x, monitor[0].y)
    screen_corners[1] = (monitor[0].x + monitor[0].width, monitor[0].y)
    screen_corners[2] = (monitor[0].x, monitor[0].y + monitor[0].height)
    screen_corners[3] = (monitor[0].x + monitor[0].width, monitor[0].y + monitor[0].height)

    # Make video frame folders for each video within temporary main folder
    # Resize and place windows in each corner of screen

    os.makedirs("_temp")
    for index,video in enumerate(videos):
        video_folder = ""
        # Remove spaces in video filename
        if(" " in video_names[index]):
            video_folder = video_names[index].replace(" ", "")
        else:
            video_folder = video_names[index]
        os.makedirs("_temp/" + video_folder)
        cv2.resizeWindow(video, monitor[0].width//2, (int) (monitor[0].height//2.25))
        cv2.moveWindow(video, screen_corners[index][0], screen_corners[index][1])
        
    done = False

    # Loop through frames in capture object
    while True:
        for index, cap in enumerate(captures):
            # Read if frame is returned and read frame itself
            ret[index],frames[index] = cap.read()
   
            # Check if video has ended or no frame is read
            if not ret[index]:
                done = True
                break

            frame_count[index] += 1

        if done:
            break
            
        for index, frame in enumerate(frames):
            # Detect ball if ball is not detected yet OR not already tracking ball
            if not ball_Detected[index]:  
                ball_Detected[index] = detect_ball(frame,index)
                Mode[index] = "Detecting"
                
            else:
                # Track ball otherwise
                if not Tracking[index]:
                    # If tracker is not initialized yet, initialize it to detected box
                    x = ball_Detected[index].origin_x
                    y = ball_Detected[index].origin_y
                    width = ball_Detected[index].width
                    height = ball_Detected[index].height

                    bbox = (x,y,width,height)
                    trackers[index].init(frame, bbox)
                    Tracking[index] = True
                else:
                    Mode[index] = "Tracking"
                    # If tracker is initialized, update tracker
                    if not track_ball(frame,index):
                        # If tracker fails, switch back to detecting
                        ball_Detected[index] = False
                        Tracking[index] = False

            # Draw camera number and mode on screen
            cv2.putText(frame,str(index+1),TOP_LEFT,FONT,FONT_SIZE,COLOR_MAGENTA,LINE_THICKNESS,cv2.LINE_AA)
            cv2.putText(frame,Mode[index],(10,monitor[0].height),FONT,FONT_SIZE,COLOR_MAGENTA,LINE_THICKNESS,cv2.LINE_AA)

            cv2.imshow(videos[index],frame)

            # Write frames to temp folder
            cv2.imwrite("_temp/" + video_names[index]+ "/frame%d.png" %frame_count[index],frame)

            
        # Wait for user to press key
        key_press = cv2.waitKey(1)

        if key_press == NO_KEY: # Keep looping if no key is pressed
            continue
        elif key_press == ESC_KEY: # Close video after ESC key is pressed
            done = True
        elif key_press == 49: # Switch capture 1 to detect after 1 is pressed
            ball_Detected[0] = False
            Tracking[0] = False
        elif key_press == 50: # Switch capture 2 to detect after 2 is pressed
            ball_Detected[1] = False
            Tracking[1] = False
        elif key_press == 51: # Switch capture 3 to detect after 3 is pressed
            ball_Detected[2] = False
            Tracking[2] = False
        elif key_press == 52: # Switch capture 4 to detect after 4 is pressed
            ball_Detected[3] = False
            Tracking[3] = False
    
    # Destroy video capture object and windows after done
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    if len(sys.argv) == 3:
        videos = os.listdir(sys.argv[1])
        paths = []
        video_names = []
        
        for video in videos:
            video_names.append(video)
            path = sys.argv[1] + "/" + video
            paths.append(path)

        output_folder = sys.argv[2]

        process_videos([paths,output_folder])
        save_output(output_folder,video_names)
    else:
        print("Usage for saving video: python3 track_ball.py <video_folder_path> <output_folder>")
        sys.exit(-1)

    
