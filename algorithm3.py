# currently the best algorithm
# uses a color range to find the ball 
# which can be adjusted

import cv2
import numpy as np

def track_soccer_ball(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Define the color range for the ball (adjust these values based on your soccer ball color)
    lower_color = np.array([25, 50, 50])
    upper_color = np.array([35, 255, 255])

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask using the specified color range
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are found, find the contour with the maximum area (likely the ball)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            
            # Get the centroid of the contour
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw a circle at the centroid
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('Soccer Ball Tracking', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

# Replace 'path/to/your/video.mp4' with the actual path to your video file
video_path = 'Soccer2.mov'
track_soccer_ball(video_path)
