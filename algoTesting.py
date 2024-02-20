import cv2
import numpy as np
from mediapipe_model_maker import object_detector

# Load the trained model
model = object_detector.ObjectDetector.load(model_path='saved_model')

# Function to detect soccer balls in a frame
def detect_soccer_balls(frame):
    # Convert frame to RGB (the model expects RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform object detection
    results = model.detect(rgb_frame)
    # Extract detection results
    for obj in results:
        if obj.label_id == 0:  # Assuming soccer ball has label_id 0
            # Get bounding box coordinates
            ymin, xmin, ymax, xmax = obj.bounding_box.flatten().tolist()
            # Convert coordinates to pixel values
            left, right, top, bottom = int(xmin * frame.shape[1]), int(xmax * frame.shape[1]), \
                                        int(ymin * frame.shape[0]), int(ymax * frame.shape[0])
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Add label
            cv2.putText(frame, 'Soccer Ball', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Function to process video
def process_video(video_path):
    # Open video capture object
    cap = cv2.VideoCapture(video_path)
    # Loop through frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Detect soccer balls in the frame
        frame = detect_soccer_balls(frame)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release video capture object
    cap.release()
    cv2.destroyAllWindows()

# Run the function with the path to the input video
process_video('input.mp4')
