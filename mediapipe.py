import os
import json
import cv2
import numpy as np
import tensorflow as tf
from mediapipe_model_maker import object_detector
from mediapipe_model_maker import face_stylizer

# Suppress TensorFlow logging messages
tf.get_logger().setLevel('ERROR')

def preprocess_image(image):
    # Resize the image to match the input size expected by the model
    resized_image = cv2.resize(image, (input_width, input_height))
    # Normalize pixel values to range [0, 1]
    normalized_image = resized_image / 255.0
    # Add batch dimension
    input_tensor = np.expand_dims(normalized_image, axis=0)
    return input_tensor

def visualize_image(image, boxes, scores):
    height, width, _ = image.shape
    for box, score in zip(boxes, scores):
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f'Score: {score:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imshow('Soccer Ball Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the trained model
model = tf.saved_model.load('retrained_model2/saved_model')

# Load an image
image = cv2.imread('path_to_your_image.jpg')

# Preprocess the image (resize, normalize, etc.)
input_height, input_width = 224, 224
input_tensor = preprocess_image(image)

# Perform inference
output_dict = model(input_tensor)

# Extract predictions (bounding boxes, scores, classes, etc.)
bounding_boxes = output_dict['detection_boxes']
scores = output_dict['detection_scores']
classes = output_dict['detection_classes']

# Filter out predictions for soccer balls (class id might vary depending on your dataset)
soccer_ball_class_id = 1
soccer_ball_indices = np.where(classes == soccer_ball_class_id)[0]

# Get bounding boxes and scores for soccer balls
soccer_ball_boxes = bounding_boxes[soccer_ball_indices]
soccer_ball_scores = scores[soccer_ball_indices]

# Visualize the detected soccer balls on the image
visualize_image(image, soccer_ball_boxes, soccer_ball_scores)
