import cv2 as cv
import sys

# Constant variables
C_BLUE = (255,0,0)
LINE_THICKNESS = 2

# Initialize tracker (can be any of the trackers offered in OpenCV)
tracker = cv.TrackerCSRT_create()

def main(argv):
    # Set video capture to file (Provide a file path in command line)
    cap = cv.VideoCapture(argv[0])

    # Display first frame of video
    ret, frame = cap.read()
    cv.imshow('Set ROI', frame)
    # User sets ROI bounding_box to a region within first frame
    bounding_box = cv.selectROI('Set ROI', frame)
    tracker.init(frame, bounding_box)

    # Video loop
    while True:
        # Read frames from video capture
        ret, frame = cap.read()
        if not ret:
            # Error: video frame not read
            break

        # Track movement within ROI per frame
        (success,box) = tracker.update(frame)
        # If there is movement, update tracking box to returned box
        if success:
            (x,y,width,height) = [int(var) for var in box]
            cv.rectangle(frame,(x,y),(x+width,y+height), C_BLUE, LINE_THICKNESS)

        # Display video frame by frame
        cv.imshow('Capture', frame)
        # Close video after ESC key is pressed
        if cv.waitKey(30) == 27:
            break
        
    # Destroy video capture object and windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])