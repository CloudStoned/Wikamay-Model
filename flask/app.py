from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import json
import pickle

app = Flask(__name__)

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Function to load the selected model and classes
def load_model_and_classes(choice):
    if choice == 'alphabet':
        model_dict = pickle.load(open('alphabet_model.p', 'rb'))
        model = model_dict['model']
        with open('ALPHABET_CLASSES.json', 'r') as json_file:
            labels_dict = json.load(json_file)
        labels_dict = {int(k): v for k, v in labels_dict.items()}
    elif choice == 'numbers':
        model_dict = pickle.load(open('numbers_model.p', 'rb'))
        model = model_dict['model']
        with open('NUMBERS_CLASSES.json', 'r') as json_file:
            labels_dict = json.load(json_file)
        labels_dict = {int(k): v for k, v in labels_dict.items()}
    else:
        model = None
        labels_dict = {}
    return model, labels_dict

# Default to alphabet recognition
model, labels_dict = load_model_and_classes('alphabet')

# Function to generate frames for video feed
def gen_frames(model, labels_dict):
    cap = cv2.VideoCapture(0)
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
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

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Get the user's choice (either 'alphabet' or 'numbers')
    choice = request.args.get('choice', 'alphabet')
    model, labels_dict = load_model_and_classes(choice)
    return Response(gen_frames(model, labels_dict), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
