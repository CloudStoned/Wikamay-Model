import os
from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64

app = Flask(__name__)

print("Starting the application...")

# Check if the model file exists
model_path = 'model.p'
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

print("Loading the model...")
try:
    # Load the model
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading the model: {str(e)}")
    exit(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a prediction request")
    # Get the image from the POST request
    image_data = base64.b64decode(request.json['image'])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        data_aux = []
        x_ = []
        y_ = []

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

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        print(f"Prediction: {predicted_character}")
        return jsonify({'prediction': predicted_character})
    else:
        print("No hand detected")
        return jsonify({'prediction': 'No hand detected'})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='192.168.1.20', port=5000, debug=True)
    print("Flask server has stopped.")