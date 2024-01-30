"""
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np


def main(argv):
    default_file = r'C:\Users\yonna\Videos\Captures\th (1).jpeg' # example file to replace
    filename = argv[0] if len(argv) > 0 else default_file
    # Determine if the input is an image or video
    is_video = filename.endswith(('.mp4', '.avi', '.mkv')) # for video choices

    if is_video:
        # If the input is a video, read from video file
        cap = cv.VideoCapture(filename)
        if not cap.isOpened():
            print('Error opening video file!')
            return -1
    else:
        # If the input is an image, read the image
        src = cv.imread(cv.samples.findFile(filename))
        # Check if image is loaded fine
        if src is None:
            print('Error opening image!')
            print('Usage: hough_lines.py [image_name or video_name -- default ' + default_file + '] \n')
            return -1

    while True:
        if is_video:
            # Read frame from video
            ret, src = cap.read() # video reading part
            if not ret:
                print('End of video')
                break

        # Convert the frame to grayscale
        if len(src.shape) == 3:
            hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
            gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        else:
            white = src
            gray = src

        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 70, 255])

        mask = cv.inRange(hsv, lower_white, upper_white)

        res = cv.bitwise_and(src, src, mask= mask)

        edges = cv.Canny(res, 50, 150)

        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(src, (x1, y1), (x2, y2), (30, 255, 30), 5)

        # Display the different types
        cv.imshow("Source", src)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", mask)
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", res)

        # Break the loop if 'Esc' key is pressed
        # only closes with the esacpe character
        key = cv.waitKey(30)
        if key == 27:
            break
        elif key == 68:  # "D" key
            cap.set(cv.CAP_PROP_POS_FRAMES, cap.get(cv.CAP_PROP_POS_FRAMES) + 120)

    if is_video:
        # Release video capture object
        cap.release()

    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
