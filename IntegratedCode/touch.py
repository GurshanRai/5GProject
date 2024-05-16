import sys
import cv2 as cv
import numpy as np

class RectangleDrawer:
    def __init__(self):
        self.rectangle_corners = []
        self.clicked = 0
        self.last_four_coords = []
        self.line_color = (142,10,234) # color is in the form of BGR

    def get_mouse_click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.rectangle_corners.append((x, y))
            self.clicked += 1
            temp_src = self.src.copy()
            if self.clicked == 2:
                cv.line(temp_src, self.rectangle_corners[0], self.rectangle_corners[1], (30, 255, 30), 5)
            elif self.clicked == 3:
                cv.line(temp_src, self.rectangle_corners[1], self.rectangle_corners[2], (30, 255, 30), 5)
            elif self.clicked == 4:
                cv.line(temp_src, self.rectangle_corners[2], self.rectangle_corners[3], (30, 255, 30), 5)
                cv.line(temp_src, self.rectangle_corners[3], self.rectangle_corners[0], (30, 255, 30), 5)
                self.last_four_coords.append(self.rectangle_corners[-4:])
                self.rectangle_corners = []
                self.clicked = 0

            self.src = temp_src
            cv.imshow('Source', self.src)

    def drawlines(self,src):
        
        points  = self.rectangle_corners
        for i in range(1,len(points)):
            cv.line(src, points[i-1], self.rectangle_corners[i], self.line_color, 5)
            if((i+1)%4==0):
                cv.line(src, points[i-3], self.rectangle_corners[i], self.line_color, 5)
        
        return src


    def process_image(self, image_path):

        if(isinstance(image_path,str)):
            self.src = cv.imread(image_path)
        else:
            self.src = image_path # detects actual image object
        
        if self.src is None:
            print('Error opening image!')
            return

        cv.namedWindow('Source')
        cv.setMouseCallback('Source', self.get_mouse_click)
        while True:
            hsv = cv.cvtColor(self.src, cv.COLOR_BGR2HSV)
            gray = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)

            lower_white = np.array([0, 0, 100])
            upper_white = np.array([180, 100, 255])

            mask = cv.inRange(hsv, lower_white, upper_white)

            res = cv.bitwise_and(self.src, self.src, mask= mask)

            cv.imshow("Source", self.src)

            key = cv.waitKey(30)
            if key == 27:
                break

        cv.destroyAllWindows()
        return self.last_four_coords

def main(image_path):
    rd = RectangleDrawer()
    return rd.process_image(image_path),rd


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python module_name.py [image_path]')
        sys.exit(1)
    main(sys.argv[1])
