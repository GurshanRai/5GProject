# Import the necessary modules.
import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constant variables for bounding box
COLOR_GREEN = (0,255,0)
BORDER_SIZE = 3

# Variables for path to model and video/image to be processed
MODEL_PATH = "retrained_model2/model.tflite"
VIDEO_PATH = "../still_samples/s.jpg"

# Adjust minimum confidence score as desired
SCORE_THRESHOLD = 0.5


# Create an ObjectDetector object from model
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=SCORE_THRESHOLD)
detector = vision.ObjectDetector.create_from_options(options)

# Load the input image/video into capture object
cap = cv2.VideoCapture(VIDEO_PATH)

# Loop through frames in capture object
while True:
    ret, frame = cap.read()

    # Convert frame to RGB format in order to be processed and detected
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(rgb_frame)

    # Draw box around detected ball if any are detected
    if len(detection_result.detections) != 0:
        x = detection_result.detections[0].bounding_box.origin_x
        y = detection_result.detections[0].bounding_box.origin_y
        width = detection_result.detections[0].bounding_box.width
        height = detection_result.detections[0].bounding_box.height

        cv2.rectangle(frame, (x, y), (x+width, y+height), COLOR_GREEN, BORDER_SIZE)

    cv2.imshow("Capture", frame)
    # Close video after ESC key is pressed
    if cv2.waitKey(30) == 27:
        break

# Destroy video capture object and windows after ESC is pressed
cap.release()
cv2.destroyAllWindows()