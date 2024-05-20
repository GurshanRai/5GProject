# Birds eye view, uses example coordinates at the moment
# Working to incorporate ball tracking coordinates 

import cv2
import numpy as np

# Load soccer field image 
soccer_field_img = cv2.imread('IntegratedCode/field.jpg')  
print("Soccer field image loaded successfully.")

# Camera calibration parameters 
focal_length = 1000  # Focal length of the camera in pixels
principal_point = (320, 240)  # Principal point (center) of the camera frame
print("Camera calibration parameters set.")

# Function to map ball position from camera frame to soccer field image
def map_ball_position(ball_position):
    # Apply geometric transformations to map ball position
    # Example: Calculate distance and angle between ball and camera center
    distance = np.linalg.norm(np.array(ball_position) - np.array(principal_point))
    angle = np.arctan2(ball_position[1] - principal_point[1], ball_position[0] - principal_point[0])
    
    # Example: Calculate corresponding position on soccer field based on distance and angle
    field_position_x = 750 * np.cos(angle)  
    field_position_y = 750 * np.sin(angle)  
    
    # Scale mapped coordinates to fit within soccer field image dimensions
    scaling_factor = 1  # Adjust this scaling factor as needed
    field_position_x_scaled = field_position_x * scaling_factor
    field_position_y_scaled = field_position_y * scaling_factor
    
    # Return scaled mapped position
    return (int(field_position_x_scaled), int(field_position_y_scaled))


# Function to detect ball in camera frame and map its position to soccer field image
def track_ball_and_map(frame, frame_count):
    # Simulate ball movement by changing ball position based on frame count
    ball_speed = 15  # Speed of ball movement (pixels per frame)
    ball_position_x = 1000 + ball_speed * frame_count  # Increasing x-coordinate for ball position
    ball_position_y = 1000 # Constant y-coordinate for ball position
    ball_position = (ball_position_x, ball_position_y)  # Ball position (x, y) in camera frame
    print("Ball detected at position: ({}, {})".format(ball_position[0], ball_position[1]))
    
    # Map ball position to soccer field image

    field_position = map_ball_position(ball_position)
    
    # Draw ball position on soccer field image
    soccer_field_with_ball = cv2.circle(soccer_field_img.copy(), field_position, 10, (0, 255, 0), -1)  # Draw green circle for ball
    print("Green circle drawn at position: ({}, {})".format(field_position[0], field_position[1]))
    
    return soccer_field_with_ball

# Main function to process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame: detect ball and map its position to soccer field image
        soccer_field_with_ball = track_ball_and_map(frame, frame_count)
        
        # Display result
        cv2.imshow('Soccer Field with Ball', soccer_field_with_ball)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

process_video('/home/parallels/Desktop/test1.mp4')  
