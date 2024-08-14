import cv2
import mediapipe as mp
import numpy as np

def extract_hand_landmarks(image_path):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.4)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return None

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Check if hand landmarks are detected
    if not results.multi_hand_landmarks:
        print("No hands detected in the image.")
        return None

    # Create a copy of the image for visualization
    annotated_image = image.copy()

    # Extract landmarks and draw them on the image
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        # Extract landmark coordinates
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })

    # Close the MediaPipe Hands object
    hands.close()

    return annotated_image, landmarks

# Example usage
image_path = 'test.jpg'  # Replace with your image path
result = extract_hand_landmarks(image_path)

if result:
    annotated_image, landmarks = result
    
    # Display the annotated image
    cv2.imshow('Hand Landmarks', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    # Print the landmarks
    print("Hand Landm   arks:")
    for i, lm in enumerate(landmarks):
        print(f"Landmark {i}: x={lm['x']:.4f}, y={lm['y']:.4f}, z={lm['z']:.4f}")
else:
    print("Failed to extract hand landmarks.")