import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

DATA_DIR = 'RFC_MODEL/data'

# Create a single figure outside the loop
plt.figure(figsize=(15, 10))

for dir_ in os.listdir(DATA_DIR):
    image_count = 0
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        if image_count >= 5:
            break

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        plt.clf()
        plt.imshow(img_rgb)
        plt.title(f"Class: {dir_}, Image: {img_path}")
        plt.axis('off')
        plt.draw()
        plt.pause(1)  # Pause for 1 second to view each image

        image_count += 1

    print(f"Finished displaying 5 images for class: {dir_}")

# Close the figure at the end
plt.close()

print("Visualization complete.")