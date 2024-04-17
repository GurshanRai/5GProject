"""
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
from line_detect_connected import remove_outliers

def main(argv):
    default_file = r'demos/My Movie 6.mov' # example file to replace
    filename = argv[0] if len(argv) > 0 else default_file
    # Determine if the input is an image or video
    is_video = filename.endswith(('.mp4', '.avi', '.mkv','.mov')) # for video choices

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
            gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        else:
            gray = src

        #inject my own code here for image processing
        
        # Apply Canny edge detection an opencv function
        edges = cv.Canny(gray, 50, 200, None, 3)

        # Copy edges to the images that will display the results in BGR
        cdst = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        # Standard Hough Line Transform
        # this is the line detection part
        lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

        # this is the part where the line is drawn for the first houghlines function
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        # Probabilistic Line finding
        linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

        # for the probablistic part
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        # Display the different types
        cv.imshow("Source", src)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

        # Break the loop if 'Esc' key is pressed
        # only closes with the esacpe character
        if cv.waitKey(30) == 27:
            break

    if is_video:
        # Release video capture object
        cap.release()

    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])