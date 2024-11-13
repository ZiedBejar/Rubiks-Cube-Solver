import cv2
import numpy as np

# Load images
img_ = cv2.imread('C:\\Users\\LENOVO\\Downloads\\img1.jpg')
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
img = cv2.imread('C:\\Users\\LENOVO\\Downloads\\img2.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()  # Changed this line

# Find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match descriptors using BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m in matches:
    if m[0].distance < 0.5 * m[1].distance:
        good.append(m)

matches = np.asarray(good)

# Check if there are enough matches
if len(matches[:, 0]) >= 4:
    src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    
    # Find homography
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    
    # Warp image
    dst_warped = cv2.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))
    
    # Blend images
    dst_warped[0:img.shape[0], 0:img.shape[1]] = img

    # Display warped and blended images using OpenCV
    cv2.imshow('Warped Image', dst_warped)
    cv2.imshow('Blended Image', dst_warped)

    # Wait for a key press and close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the blended image
    cv2.imwrite('output.jpg', dst_warped)
else:
    raise AssertionError("Can't find enough keypoints.")
