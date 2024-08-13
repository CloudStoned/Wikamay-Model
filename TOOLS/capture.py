import cv2
import mediapipe as mp
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the classes
classes = [
    "Bird", "Rabbit", "Bear", "Cow", "Cat",
    "Happy", "Sad", "Angry", "Excited", "Scared",
    "Hello", "Goodbye", "Thank you"
]

def capture_sign(sign):
    # Create directory for storing images
    save_dir = os.path.join('ASL_ANIMALS', sign)
    os.makedirs(save_dir, exist_ok=True)

    # Counter for naming files
    counter = 0
    total_images = 67
    collecting = False

    print(f"Press 's' to start/resume collecting data for '{sign}'.")
    print("Press 'p' to pause data collection.")    
    print("Press 'q' to quit data collection for this sign.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB for hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands (for visual feedback only)
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
        cv2.putText(display_frame, f"Images: {counter}/{total_images}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
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
            print(f"Quitting data collection for '{sign}'. Collected {counter} images.")
            return
        
        if collecting and counter < total_images:
            try:
                # Save the full frame
                image_path = os.path.join(save_dir, f'{sign}_{counter:03d}.jpg')
                cv2.imwrite(image_path, frame)
                
                print(f"Saved image for frame {counter+1}/{total_images}")
                counter += 1
                
                # Add a small delay to avoid duplicate frames
                time.sleep(0.1)
                
                if counter == total_images:
                    print(f"Data collection complete for '{sign}'!")
                    collecting = False
                    return
            
            except Exception as e:
                print(f"An error occurred while saving the image: {e}")
                collecting = False

# Main loop
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