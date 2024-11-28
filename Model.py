import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import model_from_json
from gtts import gTTS
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

with open('gesture_model.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights('ASL_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

labels = ['food', 'forget', 'hello', 'know', 'no', 'say', 'sky', 'thanks', 'yes']

def preprocess_landmarks(landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    landmarks = (landmarks - np.mean(landmarks)) / np.std(landmarks)
    return landmarks[:63].reshape(1, -1)  

# Text-to-Speech function
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("speech.mp3")
    os.system("start speech.mp3") 

# Open webcam feed
cap = cv2.VideoCapture(0)

last_gesture = None

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            preprocessed_landmarks = preprocess_landmarks(hand_landmarks.landmark)
            prediction = model.predict(preprocessed_landmarks)
            predicted_label = np.argmax(prediction)
            gesture = labels[predicted_label]

            if gesture != last_gesture:
                speak(gesture)
                last_gesture = gesture

    # Display the camera feed
    cv2.imshow('Hand Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
