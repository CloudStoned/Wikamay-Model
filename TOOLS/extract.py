import os
import cv2
import mediapipe as mp
import json

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Paths
data_dir = r'D:\SignLanguage\TOOLS\LETTERS_NUMS'
landmark_dir = r'D:\SignLanguage\TOOLS\LANDMARKS'

# Function to process a single image
def process_image(img_path, save_folder):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result = hands.process(image_rgb)
    landmarks = []
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
    
    if landmarks:
        landmark_file = os.path.splitext(os.path.basename(img_path))[0] + '.json'
        landmark_path = os.path.join(save_folder, landmark_file)
        with open(landmark_path, 'w') as f:
            json.dump(landmarks, f)
        print(f"Saved landmarks for image: {os.path.basename(img_path)} in {save_folder}")
    else:
        print(f"No landmarks detected for image: {os.path.basename(img_path)}")

# Loop through each folder in the dataset
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    
    # Create corresponding folder in LANDMARKS
    save_folder = os.path.join(landmark_dir, folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    print(f"Processing folder: {folder}")
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process all images in the folder
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        process_image(img_path, save_folder)
    
    print(f"Completed processing folder: {folder}")

hands.close()