import cv2
import os

DATA_DIR = './data'
if not os.path.exists(DATA_DIR) :
    os.makedirs(DATA_DIR)

gestures = int(input('enter number of gestures to input :'))
datasetSize = 200

subfolders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
subfolders.sort()
last_subfolder = int(subfolders[-1]) + 1 if subfolders else 0

gestures = gestures + last_subfolder

cap = cv2.VideoCapture(0)

for j in range(last_subfolder, gestures):
    folder_path = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print('Collecting data for gesture:', j)

    existing_files = len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0
    counter = existing_files

    while True :
        ret,frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.putText(frame,'ready ? press "Q"! ',(100,50) ,cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,0),3,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) == ord('q'):
            break

    while counter < datasetSize :
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            file_path = os.path.join(folder_path, '{}.jpg'.format(counter))
            cv2.imwrite(file_path, frame)
            counter += 1

            if cv2.waitKey(25) == ord('q'):  # Allow exit during the collection loop
                print("Exiting gesture collection early.")
                break

cap.release()
cv2.destroyAllWindows()




