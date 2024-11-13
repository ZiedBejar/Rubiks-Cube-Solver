import cv2

delay = 1
window_name = 'OpenCV QR Code'

qcd = cv2.QRCodeDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        # Display the frame
        cv2.imshow(window_name, frame)

        # Wait for a key press
        key = cv2.waitKey(delay) & 0xFF

        # Read QR code only if 'r' is pressed
        if key == ord('r'):
            ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
            if ret_qr:
                for s, p in zip(decoded_info, points):
                    if s:
                        print(s)
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    frame = cv2.polylines(frame, [p.astype(int)], True, color, 8)
            cv2.imshow(window_name, frame)

        # Exit if 'q' is pressed
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
