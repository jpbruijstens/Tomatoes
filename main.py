import cv2 as cv
import numpy as np
import sys
import os

from utils import pre_process_image, find_red_color_static

def nothing(x):
    pass

def find_green_stems(input_file, output_file, corners):
    # Read the image
    image = cv.imread(input_file)
    original_image = cv.imread(input_file)
    # Step 1: Convert image to HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Step 2: Create a mask for green color
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv.inRange(hsv_image, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=2)  # Removes noise
    mask = cv.dilate(mask, kernel, iterations=2)  # Dilation to recover the eroded main object

    # Step 3: Find contours in specific area
    if corners:
        x, y, w, h = corners
        mask = mask[y:y+h, x:x+w]
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > 2000]
        
        # Step 4: Combine contours and draw a bounding box
        if filtered_contours:
            all_contours = np.vstack([contours[i] for i in range(len(contours))])
            x, y, w, h = cv.boundingRect(all_contours)

            # Step 5: Calculate center and corner locations
            center = (x + w//2, y + h//2)
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

            # Step 6: Draw the box and show the image
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.circle(image, center, 5, (255, 0, 0), -1)

            old_box = (x, y, w, h)
            new_box = (old_box[0], old_box[1] - old_box[3]//8, old_box[2], old_box[3]//8)

            # Draw the old box
            cv.rectangle(image, (old_box[0], old_box[1]), (old_box[0] + old_box[2], old_box[1] + old_box[3]), (0, 255, 0), 2)

            # Draw the new box
            cv.rectangle(image, (new_box[0], new_box[1]), (new_box[0] + new_box[2], new_box[1] + new_box[3]), (0, 0, 255), 2)

            # Annotate center and corners on the image
            cv.putText(image, f"Center: ({center[0]}, {center[1]})", (center[0], center[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            for i, corner in enumerate(corners):
                cv.putText(image, f"C{i}: ({corner[0]}, {corner[1]})", (corner[0], corner[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
            combined_image = np.concatenate((original_image, image), axis=1)
            cv.imwrite(output_file, combined_image)



    

def find_red_tomatoes(input_file, output_file):
    corners = None
    old_box = None
    new_box = None
    # Read the image
    image = cv.imread(input_file)
    original_image = cv.imread(input_file)
    # Step 1: Convert image to HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Step 2: Create a mask for red color
    lower_red = np.array([0, 120, 130])
    upper_red = np.array([10, 255, 255])
    mask = cv.inRange(hsv_image, lower_red, upper_red)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=2)  # Removes noise
    mask = cv.dilate(mask, kernel, iterations=2)  # Dilation to recover the eroded main object

    # Step 3: Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > 2000]

    # Step 4: Combine contours and draw a bounding box
    if filtered_contours:
        all_contours = np.vstack([contours[i] for i in range(len(contours))])
        x, y, w, h = cv.boundingRect(all_contours)

        # Step 5: Calculate center and corner locations
        center = (x + w//2, y + h//2)
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

        # Step 6: Draw the box and show the image
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.circle(image, center, 5, (255, 0, 0), -1)
        old_box = (x, y, w, h)
        new_box = (old_box[0], old_box[1] - old_box[3]//8, old_box[2], old_box[3]//8)

        # Draw the old box
        cv.rectangle(image, (old_box[0], old_box[1]), (old_box[0] + old_box[2], old_box[1] + old_box[3]), (0, 255, 0), 2)

        # Draw the new box
        cv.rectangle(image, (new_box[0], new_box[1]), (new_box[0] + new_box[2], new_box[1] + new_box[3]), (0, 0, 255), 2)

        # Annotate center and corners on the image
        cv.putText(image, f"Center: ({center[0]}, {center[1]})", (center[0], center[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        for i, corner in enumerate(corners):
            cv.putText(image, f"C{i}: ({corner[0]}, {corner[1]})", (corner[0], corner[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
        combined_image = np.concatenate((original_image, image), axis=1)
        cv.imwrite(output_file, combined_image)

    return new_box

def process_all_images(input_path, output_path):
    for filename in os.listdir(input_path):
        if filename.endswith(".jpg"):
            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(output_path, filename)
            corners = find_red_tomatoes(input_file, output_file)
            # find_green_stems(input_file, output_file, corners)

            

def add_dimension_to_image(image):
    # Add an extra dimension to 'processed_image'
    processed_image = np.expand_dims(image, axis=2)

    # Repeat 'processed_image' along the third dimension
    processed_image = np.repeat(processed_image, 3, axis=2)

    return processed_image            

def process_single_image(input_file):
    if input_file.endswith(".jpg"):
        original_image = cv.imread(input_file)
        find_red_color_dynamic(original_image)
        cv.destroyAllWindows()
        sys.exit()

# create a function to find red color dynamically with the help of trackbars
def find_red_color_dynamic(original_image):
    hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
    cv.namedWindow('res')
    # Create trackbars for color change
    # Hue range is 0-179
    cv.createTrackbar('LowerH', 'res', 0, 179, nothing)
    cv.createTrackbar('LowerS', 'res', 0, 255, nothing)
    cv.createTrackbar('LowerV', 'res', 0, 255, nothing)
    cv.createTrackbar('UpperH', 'res', 179, 179, nothing)
    cv.createTrackbar('UpperS', 'res', 255, 255, nothing)
    cv.createTrackbar('UpperV', 'res', 255, 255, nothing)

    while True:
        # Get current positions of the trackbars
        l_h = cv.getTrackbarPos('LowerH', 'res')
        l_s = cv.getTrackbarPos('LowerS', 'res')
        l_v = cv.getTrackbarPos('LowerV', 'res')
        u_h = cv.getTrackbarPos('UpperH', 'res')
        u_s = cv.getTrackbarPos('UpperS', 'res')
        u_v = cv.getTrackbarPos('UpperV', 'res')   

        # Define the HSV range for red color
        lower_red = np.array([l_h, l_s, l_v])
        upper_red = np.array([u_h, u_s, u_v])

        # Threshold the HSV image to get only red colors
        mask = cv.inRange(hsv, lower_red, upper_red)

        # Bitwise-AND mask and original image
        res = cv.bitwise_and(original_image, original_image, mask=mask)

        res = pre_process_image(res)

        # combined_image = np.concatenate((original_image, res), axis=1)

        # Show the original and masked image
        cv.imshow('image', original_image)
        cv.imshow('res', res)

        k = cv.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            break

    
# main
def main(argv):
    current_dir = os.getcwd()
    input_path = os.path.join(current_dir, "images/")
    output_path = os.path.join(current_dir, "processed/")
    if argv[1] == "all":
        process_all_images(input_path, output_path)
    else:
        process_single_image(argv[1])

if __name__ == "__main__":
    main(sys.argv)