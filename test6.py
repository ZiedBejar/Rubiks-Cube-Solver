import numpy as np
import cv2
import imutils

# Function to load and resize images
def load_and_resize(image_path, width=400):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
    else:
        # Resize the image
        image = imutils.resize(image, width=width)
    return image

# Load and resize images
imageA = load_and_resize('C:\\Users\\LENOVO\\Downloads\\462540880_1090127339141460_4819507733305525814_n.jpg', width=400)  # Adjust width as needed
imageB = load_and_resize('C:\\Users\\LENOVO\\Downloads\\462557096_430788953370408_3816127114544090427_n.jpg', width=400)  # Adjust width as needed

# Check if images were loaded successfully
if imageA is not None and imageB is not None:
    # Display the images for verification
    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)

    # Create a Stitcher object
    stitcher = cv2.Stitcher_create()

    # Stitch the images together to create a panorama
    (status, result) = stitcher.stitch([imageA, imageB])

    # Check if stitching was successful
    if status == cv2.Stitcher_OK:
        print("Stitching successful")
        cv2.imshow("Result", result)
    else:
        print(f"Stitching failed with error code: {status}")
else:
    print("One or both images failed to load.")

cv2.waitKey(0)
cv2.destroyAllWindows()
