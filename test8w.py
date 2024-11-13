import numpy as np
import cv2

def load_image(image_path, width=600):
    image = cv2.imread(image_path)
    return cv2.resize(image, (500, 650))

# Load and resize images
#imageA = load_image('C:\\Users\\LENOVO\\Downloads\\462540880_1090127339141460_4819507733305525814_n.jpg')
#imageB = load_image('C:\\Users\\LENOVO\\Downloads\\462557096_430788953370408_3816127114544090427_n.jpg')

#imageA = load_image('C:\\Users\\LENOVO\\Downloads\\462110008_460753060346456_9036119171702476688_n.jpg')
#imageB = load_image('C:\\Users\\LENOVO\\Downloads\\462541988_515166297811418_2860719667566020091_n.jpg')

imageA = load_image('C:\\Users\\LENOVO\\Downloads\\img3a.jpg')
imageB = load_image('C:\\Users\\LENOVO\\Downloads\\img2a.jpg')


# Initialize SIFT detector and detect keypoints
sift = cv2.SIFT_create()
kpA, desA = sift.detectAndCompute(imageA, None)
kpB, desB = sift.detectAndCompute(imageB, None)

# Create a matcher and match descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(desA, desB)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches for visualization
match_img = cv2.drawMatches(imageA, kpA, imageB, kpB, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matches", match_img)

# Get matched keypoints
ptsA = np.float32([kpA[m.queryIdx].pt for m in matches])
ptsB = np.float32([kpB[m.trainIdx].pt for m in matches])

# Find homography and warp imageB
M, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC)
stitched = cv2.warpPerspective(imageB, M, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

# Place imageA in the stitched image
stitched[0:imageA.shape[0], 0:imageA.shape[1]] = imageA

# Show the result
cv2.imshow("Stitched Image", stitched)

cv2.waitKey(0)
cv2.destroyAllWindows()
