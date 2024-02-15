import numpy as np
import cv2 as cv

def pre_process_image(image):
    # image = cv.imread(input_file)
    # Process the image here
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (7,7), 1)
    canny_image = cv.Canny(blurred_image, 100, 100)
    dilated_image = cv.dilate(canny_image, (3,3), iterations=1)
    eroded_image = cv.erode(dilated_image, (3,3), iterations=1)
    return eroded_image

def find_red_color_static(image):
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # Define the range of red color in HSV
    lower_red = np.array([0, 180, 130])
    upper_red = np.array([10, 255, 255])
    # Threshold the HSV image to get only red colors
    mask = cv.inRange(hsv, lower_red, upper_red)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(image, image, mask=mask)
    return res
