import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from kivy.graphics.texture import Texture

# Load model and label encoder once
model = tf.keras.models.load_model('sign_language_model2.h5')
with open('label_encoder2.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

def get_frame():
    ret, frame = cap.read()
    if not ret:
        return None, None

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    prediction_text = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                X_input = np.array([landmarks])
                X_input = X_input / np.max(X_input, axis=1, keepdims=True)
                prediction = model.predict(X_input, verbose=0)
                confidence = np.max(prediction) * 100
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
                prediction_text = f'{predicted_label[0]} ({confidence:.2f}%)'

    return frame, prediction_text
