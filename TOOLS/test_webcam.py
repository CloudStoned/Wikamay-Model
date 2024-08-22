import cv2
import numpy as np

def test_webcam():
    print("Attempting to open webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam opened successfully.")
    print("Attempting to read a frame...")
    
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read a frame from the webcam.")
    else:
        print("Successfully read a frame from the webcam.")
        print(f"Frame shape: {frame.shape}")
    
    print("Releasing webcam...")
    cap.release()
    print("Webcam released.")

# Create a black image



if __name__ == "__main__":
    # test_webcam()
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Display the image
    print(cv2.__version__)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()