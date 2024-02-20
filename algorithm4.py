import cv2
import numpy as np

def track_soccer_ball(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Initialize previous frame and points
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Lucas-Kanade method
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

        # Select points that are tracked successfully
        good_new = new_points[status == 1]
        good_old = prev_points[status == 1]

        # Draw tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
            cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

        # Update previous frame and points
        prev_gray = gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)

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
