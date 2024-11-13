import cv2
import numpy as np 

# Function to adjust SIFT parameters (you can modify these values as needed)
def create_sift_detector():
    return cv2.SIFT_create(
        nfeatures=500,        # Number of features to retain
        nOctaveLayers=3,      # Number of layers in each octave
        contrastThreshold=0.04, # Contrast threshold
        edgeThreshold=10,     # Edge threshold
        sigma=1.6             # Sigma value
    )

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default camera, or you can use a video file path

# Create a SIFT detector object
sift = create_sift_detector()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the resulting frame
    cv2.imshow('SIFT Keypoints', img_with_keypoints)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
