import sys
import cv2 as cv
import numpy as np

# Global variables to store rectangle corners
rectangle_corners = []
clicked = 0
last_four_coords = []

# Mouse callback function
def get_mouse_click(event, x, y, flags, param):
    global rectangle_corners, clicked, last_four_coords, src

    if event == cv.EVENT_LBUTTONDOWN:
        rectangle_corners.append((x, y))
        clicked += 1
        temp_src = src.copy()
        if clicked == 2:
            # Draw rectangle between points 1 and 2
            cv.line(temp_src, rectangle_corners[0], rectangle_corners[1], (30, 255, 30), 5)
        elif clicked == 3:
            # Draw line between points 2 and 3
            cv.line(temp_src, rectangle_corners[1], rectangle_corners[2], (30, 255, 30), 5)
        elif clicked == 4:
            # Reset points
            cv.line(temp_src, rectangle_corners[2], rectangle_corners[3], (30, 255, 30), 5)
        elif clicked == 5:
            cv.line(temp_src, rectangle_corners[3], rectangle_corners[4], (30, 255, 30), 5)
            last_four_coords.append(rectangle_corners[-4:])
            rectangle_corners = []
            clicked = 0

        src = temp_src
        cv.imshow('Source', src)


def main(argv):
    global src

    default_file = r"C:\Users\yonna\Videos\Captures\th.jpeg"
    filename = argv[0] if len(argv) > 0 else default_file
    is_video = filename.endswith(('.mp4', '.avi', '.mkv'))

    if is_video:
        cap = cv.VideoCapture(filename)
        if not cap.isOpened():
            print('Error opening video file!')
            return -1
    else:
        src = cv.imread(cv.samples.findFile(filename))
        if src is None:
            print('Error opening image!')
            print('Usage: hough_lines.py [image_name or video_name -- default ' + default_file + '] \n')
            return -1

    cv.namedWindow('Source')
    cv.setMouseCallback('Source', get_mouse_click)
    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                print('End of video')
                break
            src = frame.copy()
        if len(src.shape) == 3:
            hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
            gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        else:
            white = src
            gray = src

        lower_white = np.array([0, 0, 100])
        upper_white = np.array([180, 100, 255])

        mask = cv.inRange(hsv, lower_white, upper_white)

        res = cv.bitwise_and(src, src, mask= mask)

        cv.imshow("Source", src)



        key = cv.waitKey(30)
        if key == 27:
            break

    if is_video:
        cap.release()

    cv.destroyAllWindows()
    print(last_four_coords)
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
