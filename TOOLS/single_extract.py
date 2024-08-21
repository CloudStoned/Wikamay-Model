import cv2
import mediapipe as mp
import numpy as np
import os

def extract_hand_landmarks(image_path):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

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

    landmarks_list = []
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
        landmarks_list.append(landmarks)

    # Close the MediaPipe Hands object
    hands.close()

    return annotated_image, landmarks_list

# Example usage
# image_path = r'D:\SignLanguage\TOOLS\C_7 (6).jpg'
image_path = r'D:\SignLanguage\MAIN_DATASET\Backup\LEFT_NAMES\CLOUD\9\9_LEFT_C (1).jpg'  # Replace with your image path
  # Replace with your image path
if not os.path.exists(image_path):
    print(f"Error: The file {image_path} does not exist.")
else:
    result = extract_hand_landmarks(image_path)

    if result:
        annotated_image, landmarks_list = result
        
        # Resize the image to 266x266
        resized_image = cv2.resize(annotated_image, (300, 300), interpolation=cv2.INTER_AREA)
        
        # Display the resized annotated image
        cv2.imshow('Hand Landmarks', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
        # Print the landmarks
        print("Hand Landmarks:")
        for hand_index, landmarks in enumerate(landmarks_list):
            print(f"Hand {hand_index + 1}:")
            for i, lm in enumerate(landmarks):
                print(f"Landmark {i}: x={lm['x']:.4f}, y={lm['y']:.4f}, z={lm['z']:.4f}")
    else:
        print("Failed to extract hand landmarks.")

# Save the annotated images
# if result:
#     cv2.imwrite('annotated_hand_original.jpg', annotated_image)
#     cv2.imwrite('annotated_hand_resized.jpg', resized_image)
#     print("Original annotated image saved as 'annotated_hand_original.jpg'")
#     print("Resized annotated image saved as 'annotated_hand_resized.jpg'")