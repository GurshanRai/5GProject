import cv2
import numpy as np
from scipy import stats

def process_image(path):
    # Load the image
    if(isinstance(path,str)):
        image = cv2.imread(path)
    else:
        image= path
        
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #test =np.average(gray_image)
    #test2= np.std(gray_image)

    #test3=np.median(gray_image)
    #int(test+test2)
    # Set the brightness range (adjust as needed)
    lower_brightness = 100
    upper_brightness = 255

    # Create a binary mask for pixels within the brightness range
    binary_mask = cv2.inRange(gray_image, lower_brightness, upper_brightness)

    # Create a black image
    result_image = np.zeros_like(image)

    # Set white color to the white areas in the result image
    result_image[binary_mask == 255] = [255, 255, 255]  # White color in RGB format

    # Save the result
    cv2.imwrite('black_white.jpg', result_image)
    return result_image

    #print(result_image)
    #print(np.size(image))
    #print(np.shape(image))

