import cv2
import mediapipe as mp
import os
import uuid

IMAGE_PATH = './CollectedImages2'
labels = input("Enter gesture labels separated by commas: ").split(",")
labels = [label.strip() for label in labels]

dataset_size = 500

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for label in labels:
    folder_path = os.path.join(IMAGE_PATH, label)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f'Get ready for "{label}"...')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.putText(frame, 'Press "Q" when ready!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    print(f'Capturing images for "{label}"...')

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            imagename = os.path.join(folder_path, f'{label}_{counter}_{uuid.uuid1()}.jpg')
            cv2.imwrite(imagename, frame)
            counter += 1

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            print("Exiting gesture collection early.")
            break

cap.release()
cv2.destroyAllWindows()