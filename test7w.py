import imutils
import numpy as np
import cv2

def load_and_resize(image_path, width=400):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
    else:
        # Resize the image
        image = imutils.resize(image, width=width)
    return image

# Load images
#imageA = load_and_resize('C:\\Users\\LENOVO\\Downloads\\462540880_1090127339141460_4819507733305525814_n.jpg', width=500)
#imageB = load_and_resize('C:\\Users\\LENOVO\\Downloads\\462557096_430788953370408_3816127114544090427_n.jpg', width=500)

imageA = load_and_resize('C:\\Users\\LENOVO\\Downloads\\img3a.jpg')
imageB = load_and_resize('C:\\Users\\LENOVO\\Downloads\\img2a.jpg')

# Check if images were loaded successfully
if imageA is None or imageB is None:
    print("Error: One or both images could not be loaded.")
else:
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
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

    # Extract matched keypoints
    ptsA = np.float32([kpA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpB[m.trainIdx].pt for m in matches])

    # Find homography
    M, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC)

    # Get dimensions of both images
    heightA, widthA = imageA.shape[:2]
    heightB, widthB = imageB.shape[:2]

    # Calculate the size of the output stitched image
    pointsA = np.array([[0, 0], [0, heightA - 1], [widthA - 1, heightA - 1], [widthA - 1, 0]]).astype(float)
    pointsB = np.array([[0, 0], [0, heightB - 1], [widthB - 1, heightB - 1], [widthB - 1, 0]]).astype(float)

    # Transform points from image B to the perspective of image A
    transformed_pointsB = cv2.perspectiveTransform(pointsB.reshape(-1, 1, 2), M)

    # Squeeze the transformed points to remove the extra dimension
    transformed_pointsB = transformed_pointsB.squeeze()

    # Get the bounding box of both images
    all_points = np.concatenate((pointsA, transformed_pointsB), axis=0)
    [x_min, y_min] = np.int32(all_points.min(axis=0) - 0.5)
    [x_max, y_max] = np.int32(all_points.max(axis=0) + 0.5)

    # Calculate the translation
    translation_dist = [-x_min, -y_min]

    # Create the transformation matrix
    M_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Warp image B with the translation
    stitchedB = cv2.warpPerspective(imageB, M_translation.dot(M), (x_max - x_min, y_max - y_min))
    
    # Place image A in the final stitched image
    stitchedA = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    stitchedA[translation_dist[1]:translation_dist[1] + heightA, translation_dist[0]:translation_dist[0] + widthA] = imageA

    # Blend both images
    stitched_result = cv2.addWeighted(stitchedA, 0.5, stitchedB, 0.5, 0)

    # Show the result
    cv2.imshow("Stitched Image", stitched_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
