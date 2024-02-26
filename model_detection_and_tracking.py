# Import the necessary modules.
import sys
import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constant variables
# Variables for bounding box color and thickness
COLOR_GREEN = (0,255,0)
LINE_THICKNESS = 3
# Variables for path to model and video/image to be processed
MODEL_PATH = ""
VIDEO_PATH = ""
# Adjust minimum confidence score as desired
SCORE_THRESHOLD = 0.55

# Create an ObjectDetector object from model
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=SCORE_THRESHOLD)

# Initialize detector from ObjectDetector object
detector = vision.ObjectDetector.create_from_options(options)
# Initialize tracker (can be any of the trackers offered in OpenCV)
tracker = cv2.TrackerCSRT_create()

def detect_ball(frame):
    # Convert frame to RGB format in order to be processed and detected
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(rgb_frame)

    # Draw box around detected ball if any are detected
    if len(detection_result.detections) != 0:
        return detection_result.detections[0].bounding_box
    
    cv2.imshow("Capture", frame)
    return None
    

def track_ball(frame):
    # Track movement within ROI per frame
    (success,box) = tracker.update(frame)
    # If there is movement, update tracking box to returned box
    if success:
        (x,y,width,height) = [int(var) for var in box]
        cv2.rectangle(frame,(x,y),(x+width,y+height), COLOR_GREEN, LINE_THICKNESS)
        cv2.imshow("Capture", frame)
        return True
    return False

# Load the input image/video into capture object
cap = cv2.VideoCapture(VIDEO_PATH)

ball_detected = False
tracking = False

# Loop through frames in capture object
while True:
    ret,frame = cap.read()

    if not ret:
        break

    if not ball_detected:
        # Detect ball if not already tracking ball
        ball_detected = detect_ball(frame)
    else:
        # Track ball otherwise
        if not tracking:
            # initialize tracker
            x = ball_detected.origin_x
            y = ball_detected.origin_y
            width = ball_detected.width
            height = ball_detected.height

            bbox = (x,y,width,height)
            tracker.init(frame, bbox)
            tracking = True
        else:
            if not track_ball(frame):
                ball_detected = False
                tracking = False
            
    key_press = cv2.waitKey(1)

    if key_press == -1:
        continue
    elif key_press == 27: # Close video after ESC key is pressed
        break
    elif key_press == 32: # Switch to detect after SPACE is pressed
        ball_detected = False
        tracking = False
        

# Destroy video capture object and windows after ESC is pressed
cap.release()
cv2.destroyAllWindows()