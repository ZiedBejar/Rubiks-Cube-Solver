import cv2
import numpy as np

# Function to detect SIFT keypoints and descriptors
def detect_and_compute(image, sift):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# Load two images
image1_path = 'C:\\Users\\LENOVO\\Downloads\\img1.jpg'
image2_path = 'C:\\Users\\LENOVO\\Downloads\\img2.jpg'
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Resize images if needed (optional)
image1 = cv2.resize(image1, (600, 400))  # Example resizing
image2 = cv2.resize(image2, (600, 400))

# Create a SIFT detector object
sift = cv2.SIFT_create()

# Detect keypoints and descriptors in both images
keypoints1, descriptors1 = detect_and_compute(image1, sift)
keypoints2, descriptors2 = detect_and_compute(image2, sift)

# Create a BFMatcher object for feature matching
bf = cv2.BFMatcher()

# Match descriptors between the two images
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test as per Lowe's paper to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Get the matching keypoints for both images
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Estimate homography matrix to warp one image onto another
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp image1 to align with image2
height, width, channels = image2.shape
warped_image1 = cv2.warpPerspective(image1, H, (width, height))

# Blend the two images (simple averaging here)
blended_image = cv2.addWeighted(warped_image1, 0.5, image2, 0.5, 0)

# Show the result
cv2.imshow('img1',warped_image1)
cv2.imshow('img2',image2)
cv2.imshow('Blended Image', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
