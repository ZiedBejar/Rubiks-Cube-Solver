import cv2
import numpy as np

# Function to create a SIFT detector object
def create_sift_detector():
    return cv2.SIFT_create(
        nfeatures=20,         # Number of features to retain
        nOctaveLayers=3,       # Number of layers in each octave
        contrastThreshold=0.04, # Contrast threshold
        edgeThreshold=15,      # Edge threshold
        sigma=2             # Sigma value
    )

# Function to detect and compute SIFT keypoints and descriptors
def detect_and_compute(image, sift):
    if image is None:
        raise ValueError("Image is None, check if the image was loaded correctly.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# Function to resize images or frames while maintaining aspect ratio
def resize_image(image, target_width):
    (h, w) = image.shape[:2]
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    return cv2.resize(image, (target_width, target_height))

# Load reference image
reference_image_path = 'C:\\Users\\LENOVO\\Pictures\\Camera Roll\\WIN_20240925_14_37_32_Pro.jpg'
reference_image = cv2.imread(reference_image_path)

# Check if the reference image was loaded correctly
if reference_image is None:
    raise FileNotFoundError(f"Unable to load image at path: {reference_image_path}")

# Resize the reference image to 400 pixels wide
reference_image_resized = resize_image(reference_image, 400)

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

# Create SIFT detector object
sift = create_sift_detector()

# Detect features in reference image
reference_keypoints, reference_descriptors = detect_and_compute(reference_image_resized, sift)

# Create a BFMatcher object for feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

paused = False  # Flag to pause the video

while True:
    if not paused:  # Only grab a new frame if not paused
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize the video frame to match the reference image width (400 pixels)
        frame_resized = resize_image(frame, 400)

        # Detect features in the current video frame
        frame_keypoints, frame_descriptors = detect_and_compute(frame_resized, sift)

        if frame_descriptors is not None:  # Check if descriptors were found in the frame
            # Match descriptors between reference image and video frame
            matches = bf.knnMatch(reference_descriptors, frame_descriptors, k=2)

            # Apply ratio test as per Lowe's paper
            good_matches = []
            for m, n in matches:
                if m.distance < 0.50 * n.distance:
                    good_matches.append(m)

            # Draw matches on the frame
            img_matches = cv2.drawMatches(reference_image_resized, reference_keypoints, frame_resized, frame_keypoints, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Display the results in a single window
            cv2.imshow('SIFT Matching', img_matches)
        else:
            # Show the frame without matching if descriptors were not found
            combined_image = np.hstack((reference_image_resized, frame_resized))
            cv2.imshow('SIFT Matching', combined_image)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Exit on 'q' key
        break
    elif key == ord(' '):  # Toggle pause/play on spacebar
        paused = not paused

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
