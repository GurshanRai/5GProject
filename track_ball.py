# Import the necessary modules.
# This program only requires "pip install mediapipe" to run
import sys
import time
import os
import cv2
import threading

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Constant variables
# Variables for key presses
NO_KEY = -1
ESC_KEY = 27
SPACE_KEY = 32
# Variables for drawing text/boxes
COLOR_GREEN = (0,255,0)
LINE_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SIZE = 3
TOP_LEFT = (15,40)
BOTTOM_RIGHT = (5, 1070)

# Variable for path to model
MODEL_PATH = "custom_model/model/model.tflite"
# Adjust minimum confidence score as desired
SCORE_THRESHOLD = 0.55

# Create an ObjectDetector object from model
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=SCORE_THRESHOLD)

# Initialize detector from ObjectDetector object
detectors = []
for _ in range(4):
    detector = vision.ObjectDetector.create_from_options(options)
    detectors.append(detector)

# Initialize tracker (can be any of the trackers offered in OpenCV)
trackers = []
for _ in range(4):
    tracker = cv2.TrackerCSRT_create()
    trackers.append(tracker)


def detect_ball(frame, index):
    # Convert frame to RGB format in order to be processed and detected
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detectors[index].detect(rgb_frame)

    # Draw box around detected ball if any are detected
    if len(detection_result.detections) != 0:
        return detection_result.detections[0].bounding_box
    
    return None
    

def track_ball(frame,index):
    # Track movement within ROI per frame
    (success,box) = trackers[index].update(frame)
    # If there is movement, update tracking box to returned box
    if success:
        (x,y,width,height) = [int(var) for var in box]
        cv2.rectangle(frame,(x,y),(x+width,y+height), COLOR_GREEN, LINE_THICKNESS)

        return True
    return False

def main(argv):
    # FPS display variables
    # Time of previous frame and time of new frame
    prev_frame = 0
    new_frame = 0
    frame_count = 0
    
    # If no cmd line arguments given, use video camera
    #video_camera = True

    video_camera = False
    videos = argv

    captures = []
    for video in videos:
        cv2.namedWindow(video,cv2.WINDOW_NORMAL)
        # Load the input image/video into capture object
        cap = cv2.VideoCapture(video)
        captures.append(cap)

    if not cap.isOpened():
        print("Camera could not be opened\n")

    ball_detected = False
    tracking = False

    mode = ""
    ret = [None] * len(videos)
    frames = [None] * len(videos)
    ball_Detected = [False] * len(videos)
    Tracking = [False] * len(videos)
    Mode = [""] * len(videos)

    # Loop through frames in capture object
    while True:
        for index, cap in enumerate(captures):
            # Read if frame is returned and read frame itself
            ret[index],frames[index] = cap.read()
            '''
            Code for taking streams
            # Check if video has ended or if connection is lost
            # If using video camera, retry connection
            if ret[index]:
                if video_camera:
                    cap = cv2.VideoCapture(videos)
                    print("Attempting to reconnect...")
                    time.sleep(2)
                    continue
                else:
                    # Otherwise, no ret means video has ended
                    sys.exit(1)
                '''
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

            '''
            # Calculate and display FPS
            new_frame = time.time()
            fps = 1 / (new_frame-prev_frame)
            prev_frame = new_frame

            fps = (int)(fps)
            fps = (str)(fps)
            '''
            cv2.putText(frame,str(index+1),TOP_LEFT,FONT,FONT_SIZE,COLOR_GREEN,LINE_THICKNESS,cv2.LINE_AA)
            cv2.putText(frame,Mode[index],BOTTOM_RIGHT,FONT,FONT_SIZE,COLOR_GREEN,LINE_THICKNESS,cv2.LINE_AA)
            cv2.imshow(videos[index],frame)
        # Wait for user to press key
        key_press = cv2.waitKey(1)

        if key_press == NO_KEY: # Keep looping if no key is pressed
            continue
        elif key_press == ESC_KEY: # Close video after ESC key is pressed
            sys.exit(1)
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
    

        '''
        # Write frames to output folder
        if not video_camera:  
            cv2.imwrite(out + "frame%d.png" %frame_count, frame)
        frame_count+=1
        '''

        
        '''
        # Stitch spliced video frames into one video
        if not video_camera:
            images = [img for img in os.listdir(output_dir) if img.endswith(".png")]
            frame = cv2.imread(os.path.join(output_dir, images[0]))
            height, width, layers = frame.shape

            video = cv2.VideoWriter("video.avi", 0, 60, (width,height))

            for image in images:
                video.write(cv2.imread(os.path.join(output_dir, image)))
      
      '''
    for cap in captures:
        # Destroy video capture object and windows after ESC is pressed
        cap.release()
    #video.release()

    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    '''
    Code for taking streams
    # If no cmd line arguments given, use GoPro streams
    if len(sys.argv) == 1:
        # Initialize camera RTMP addresses for threading
        cameras = ["rtmp://192.168.1.72/live/gp1",
                "rtmp://192.168.1.72/live/gp2",
                "rtmp://192.168.1.72/live/gp3",
                "rtmp://192.168.1.72/live/gp4"]   
        for camera in cameras:
            th = threading.Thread(target=main, args=[camera])
            th.daemon = True
            th.start()
    '''
    # Otherwise, use argument as video file path
    if len(sys.argv) == 2:
        videos = os.listdir(sys.argv[1])
        paths = []
        for video in videos:
            path = sys.argv[1] + "/" + video
            paths.append(path)
            
        main(paths)
    else:
        print("Usage for GoPro stream: python3 track_ball.py")
        print("Usage for saving video: python3 track_ball.py <video_folder_path>")
        sys.exit(-1)

    # ffmpeg -framerate 60 -i frame%d.png -c:v libx264 -r 60 output.mp4