import cv2
from pyzbar import pyzbar

def detect_and_decode_barcodes(frame):
    # Detect barcodes and QR codes in the frame
    barcodes = pyzbar.decode(frame)
    
    for barcode in barcodes:
        # Extract the bounding box location of the barcode and draw a rectangle around it
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the barcode data
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        
        # Display the barcode data and type on the frame
        text = "{} ({})".format(barcode_data, barcode_type)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect and decode barcodes and QR codes
        frame = detect_and_decode_barcodes(frame)
        
        # Display the resulting frame
        cv2.imshow('Barcode and QR Code Detection', frame)
        
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
