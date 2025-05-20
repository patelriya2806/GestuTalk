import cv2
import mediapipe as mp
import os
import pandas as pd

IMAGE_PATH = './CollectedImages2'
OUTPUT_CSV = './sign_language_landmarks2.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

data = []

for label in os.listdir(IMAGE_PATH):
    folder_path = os.path.join(IMAGE_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    print(f'Processing images for "{label}"...')

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read {image_file}, skipping...")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_data = [label]
            for lm in hand_landmarks.landmark:
                hand_data.extend([lm.x, lm.y, lm.z])
            data.append(hand_data)

columns = ['Label'] + [f'{axis}{i}' for i in range(21) for axis in ['X', 'Y', 'Z']]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("Landmark dataset created successfully!")
