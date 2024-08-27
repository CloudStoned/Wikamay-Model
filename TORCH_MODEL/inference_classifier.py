import pickle
import cv2
import mediapipe as mp
import numpy as np
import torch
import os

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the model architecture
class HandGestureModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandGestureModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(input_size, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(20, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load the PyTorch model
model_path = r'D:\SignLanguage\TORCH_MODEL\hand_gesture_model.pt'
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    exit()

try:
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Create a new instance of the model
    input_size = 42  # 21 landmarks * 2 (x and y)
    num_classes = len(state_dict['model.6.weight'])  # Get number of classes from the last layer
    model = HandGestureModel(input_size, num_classes)
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Load the label mapping
label_mapping_path = r'D:\SignLanguage\TORCH_MODEL\label_mapping.pickle'
if not os.path.exists(label_mapping_path):
    print(f"Error: Label mapping file {label_mapping_path} not found.")
    exit()

try:
    with open(label_mapping_path, 'rb') as f:
        label_to_id = pickle.load(f)
        id_to_label = {v: k for k, v in label_to_id.items()}
    print("Label mapping loaded successfully")
except Exception as e:
    print(f"Error loading the label mapping: {e}")
    exit()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            # Prepare input for PyTorch model
            input_tensor = torch.tensor([data_aux], dtype=torch.float32).to(device)
            
            # Get prediction from PyTorch model
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_label = id_to_label[predicted.item()]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            print(f"Predicted label: {predicted_label}")
        except Exception as e:
            print(f"Error during inference: {e}")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()