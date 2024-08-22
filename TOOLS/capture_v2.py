import cv2
import mediapipe as mp
import os
import time
import json
import traceback

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)
NAME_LETTER = "C"

# Define the classes
# Define the classes
classes = [
    "Bird", "Rabbit", "Bear", "Cow", "Cat",
    "Happy", "Sad", "Angry", "Excited", "Scared",
    "Hello", "Goodbye", "Thank you"
]

def capture_sign(sign):
    print(f"Entering capture_sign function for sign: {sign}")
    try:
        # Create directory structure for storing images and landmarks
        base_dir = os.path.join('TOOLS', 'GESTURES', sign)
        img_dir = os.path.join(base_dir, 'IMG')
        landmark_dir = os.path.join(base_dir, 'LANDMARK')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(landmark_dir, exist_ok=True)
        print(f"Directories created: {img_dir}, {landmark_dir}")

        # Counter for naming files
        counter = 1
        total_images = 67
        collecting = False

        print(f"Press 's' to start/resume collecting data for '{sign}'.")
        print("Press 'p' to pause data collection.")    
        print("Press 'q' to quit data collection for this sign.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Convert the BGR image to RGB for hand detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = hands.process(rgb_frame)
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Draw hand landmarks on the display frame (for visual feedback)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Add status text to the display frame
            status_text = "Collecting" if collecting else "Paused"
            cv2.putText(display_frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if collecting else (0, 0, 255), 2)
            cv2.putText(display_frame, f"Images: {counter-1}/{total_images}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Sign: {sign}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display the frame with landmarks and status
            cv2.imshow('Sign Language Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                collecting = True
                print(f"Data collection started/resumed. Show sign for '{sign}'.")
            elif key == ord('p'):
                collecting = False
                print("Data collection paused.")
            elif key == ord('q'):
                print(f"Quitting data collection for '{sign}'. Collected {counter-1} images.")
                return
            
            if collecting and counter <= total_images:
                try:
                    # Save the full frame
                    image_path = os.path.join(img_dir, f'{NAME_LETTER}_{counter}.jpg')
                    cv2.imwrite(image_path, frame)
                    
                    # Extract and save landmarks
                    if results.multi_hand_landmarks:
                        landmarks_data = []
                        for hand_landmarks in results.multi_hand_landmarks:
                            hand_data = []
                            for landmark in hand_landmarks.landmark:
                                hand_data.append({
                                    'x': landmark.x,
                                    'y': landmark.y,
                                    'z': landmark.z
                                })
                            landmarks_data.append(hand_data)
                        
                        # Save landmarks as JSON
                        landmarks_path = os.path.join(landmark_dir, f'{NAME_LETTER}_{counter}_landmarks.json')
                        with open(landmarks_path, 'w') as f:
                            json.dump(landmarks_data, f)
                    
                    print(f"Saved image and landmarks for frame {counter}/{total_images}")
                    counter += 1
                    
                    # Add a small delay to avoid duplicate frames
                    time.sleep(0.1)
                    
                    if counter > total_images:
                        print(f"Data collection complete for '{sign}'!")
                        collecting = False
                        return
                
                except Exception as e:
                    print(f"An error occurred while saving the data: {e}")
                    print(traceback.format_exc())
                    collecting = False

    except Exception as e:
        print(f"An error occurred in capture_sign function: {e}")
        print(traceback.format_exc())

while True:
    print("\nAvailable classes:")
    for i, cls in enumerate(classes, 1):
        print(f"{i}. {cls}")
    print("Type 'quit' to exit the program.")
    
    user_input = input("Enter the number or name of the class to capture (or 'quit' to exit): ").strip()
    
    if user_input.lower() == 'quit':
        break
    
    if user_input.isdigit():
        index = int(user_input) - 1
        if 0 <= index < len(classes):
            selected_class = classes[index]
        else:
            print(f"Invalid number. Please enter a number between 1 and {len(classes)}.")
            continue
    elif user_input in classes:
        selected_class = user_input
    else:
        print("Invalid input. Please enter a valid number or class name.")
        continue
    
    print(f"Selected class: {selected_class}")
    capture_sign(selected_class)

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

print("Script ended.")