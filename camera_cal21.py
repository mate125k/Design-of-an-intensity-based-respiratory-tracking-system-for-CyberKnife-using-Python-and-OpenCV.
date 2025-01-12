import numpy as np
import cv2 as cv
import glob  # For reading multiple images from a folder
import os 


# Define the chessboard dimensions (number of internal corners per row and column)
chessboard_size = (8, 6)  # adjust to match your chessboard pattern
FrameSize=(3840 ,2748)

# Prepare object points like (0,0,0), (1,0,0), (2,0,0), ..., scaled by square size
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)


# Arrays to store object points and image points for all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


directory_path= 'C:\\7_felev\\szakdolgozat_program\\proba' # innnen veszem a képet ami fel akarok dolgozni
output_dir = "C:\\7_felev\\szakdolgozat_program\\solved" # A képek új mappába keruljenek a kocka sarka keresés


###### összegyűjtjük az összes fájlt ha TIFFT AKARSZ IT KELL MEGVÁLTOZTATNI #################
images = glob.glob(f'{directory_path}/*.png') # még nem rajzolt kép
image_solved = glob.glob(f'{output_dir}/*.png')# már rajzolt kép


def display_resized_image(img, window_name="Image", max_width=800, max_height=600):
    
    # Get the original dimensions
    height, width = img.shape[:2]

    # Calculate the scaling factor to fit within max dimensions
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)

    # Resize the image if necessary
    if scale < 1:  # Only scale down if the image is larger than max dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)

    # Display the resized image
    cv.imshow(window_name, img)
    cv.waitKey(1000)
    cv.destroyAllWindows()

def calculate_reprojection_error(objPoints, imgPoints, rvecs, tvecs, cameraMatrix, dist):
    
   mean_error = 0
   for i in range(len(objPoints)):
     imgPoints2, _ = cv.projectPoints(objPoints[i],rvecs[i], tvecs[i], cameraMatrix ,dist )
     error = cv.norm(imgPoints[i],imgPoints2, cv.NORM_L2)/ len(imgPoints2)
     mean_error += error

   print("\n total error: {}".format( mean_error/len(objPoints)))

def undistort_images(image_paths, cameraMatrix, dist, undistorted_dir):
    # Ensure the output directory exists
    if not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)
    
    for i, image_path in enumerate(image_paths):
        # Read the image
        img = cv.imread(image_path)
        
        # Check if image loading was successful
        if img is None:
            print(f"Failed to load image at {image_path}")
            continue
        
        # Get the dimensions of the image
        h, w = img.shape[:2]

        # Compute the new camera matrix
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

        # Undistort the image
        dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

        # Crop the image to the valid region of interest
        x, y, roi_w, roi_h = roi
        dst = dst[y:y+roi_h, x:x+roi_w]

        # Define the output filename
        output_filename = os.path.join(undistorted_dir, f"undistorted_image_{i}.png")

        # Save the undistorted image
        cv.imwrite(output_filename, dst)
        print(f"Saved undistorted image as {output_filename}")

        # Optionally display the resized image
        display_resized_image(dst, "Undistortion")

# Define a function to display images with optional resizing
def display_resized_image(img, window_name, max_width=800, max_height=600):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1)  # Scale down if the image is too large
    if scale < 1:
        img = cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    cv.imshow(window_name, img)
    cv.waitKey(500)  # Display each image for 500 ms
    cv.destroyWindow(window_name)


def find_and_save_chessboard_corners_and_calibrate(image_paths, chessboard_size, criteria, output_dir, frame_size):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for i, image_path in enumerate(image_paths):
        # Read the image
        img = cv.imread(image_path)
        if img is None:
            print(f"Could not load image at {image_path}")
            continue

        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        # If corners are found, refine them and add to points list
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw the chessboard corners on the image
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)

            # Define the output filename
            output_filename = os.path.join(output_dir, f"solved_image_{i}.png")

            # Save the image with drawn corners
            cv.imwrite(output_filename, img)
            print(f"Saved image with detected corners as {output_filename}")

            # Optionally display the resized image
            display_resized_image(img, "Detected Corners")

    # Camera Calibration
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    # Print the calibration results
    print("Camera Calibrated:", ret)
    print("\nCamera Matrix:\n", cameraMatrix)
    print("\nDistortion Parameters:\n", dist)
    print("\nRotation Vectors:\n", rvecs)
    print("\nTranslation Vectors:\n", tvecs)

    return objpoints, imgpoints, cameraMatrix, dist, rvecs, tvecs






#### chessboard corner searching and Calibration ##########################


find_and_save_chessboard_corners_and_calibrate(images, chessboard_size, criteria,output_dir,FrameSize)


#### Undistortion ##################################################

undistorted_dir = "C:/7_felev/szakdolgozat_program/undistorted"

undistort_images(image_solved, cameraMatrix, dist, undistorted_dir)

### Reprojection error ######x

calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, cameraMatrix, dist)





