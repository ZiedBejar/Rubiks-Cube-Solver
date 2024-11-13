import cv2
import numpy as np

# Parameters
qr_code_size_cm = 4  # Known size of the QR code (4 cm)

# Function to calculate the black rectangle dimensions
def calculate_black_bar_dimensions(frame, points):
    # Draw the QR code outline
    for p in points:
        p = p.astype(int)
        cv2.polylines(frame, [p], True, (0, 255, 0), 8)

        # Calculate QR code size in pixels
        width_px = np.linalg.norm(p[0] - p[1])  # Horizontal width in pixels

        # Calculate the scale factor (cm per pixel)
        scale_factor = qr_code_size_cm / width_px

        # Convert the image to grayscale and apply threshold for contour detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours to detect the black rectangle
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by area and shape (assuming the black rectangle is large and horizontal)
            aspect_ratio = float(w) / h
            if aspect_ratio > 3 and w > 50:  # Adjust the thresholds if necessary
                # Calculate black rectangle dimensions in centimeters
                black_bar_width_cm = w * scale_factor
                black_bar_height_cm = h * scale_factor

                # Draw the rectangle outline
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

                print(f"Black Bar Dimensions: {black_bar_width_cm:.2f} cm x {black_bar_height_cm:.2f} cm")

    return frame

# Open the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect QR code
    qr_detector = cv2.QRCodeDetector()
    ret_qr, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)

    # If QR code is detected, calculate black rectangle dimensions
    if ret_qr:
        frame = calculate_black_bar_dimensions(frame, points)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Wait for 'c' key to capture the frame
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # If QR code detected, calculate and display the dimensions
        if ret_qr:
            print("QR Code detected. Processing dimensions...")

    # Break the loop with 'q' key
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
