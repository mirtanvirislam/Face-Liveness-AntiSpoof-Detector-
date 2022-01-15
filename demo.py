from SpoofDetector import *
import torch
import cv2
import dlib
import time
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt


def distance(x1, y1, x2, y2):
    # Returns Euclidean distance between two points
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


cap = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
spoof_detector = SpoofDetector()  # SpoofDetector class

roe_list = []
error_list = []
frame_count = 0
tic = time.perf_counter()
error_log_file = open('error_log.csv', mode='w', newline="")
csv_writer = csv.writer(error_log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Start camera stream loops
while True:
    try:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            roe_list = []

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_frame = frame[y1 - 50:y2, x1 - 50:x2, :]
            landmarks = predictor(gray, face)
            blob = torch.tensor(cv2.dnn.blobFromImage(cv2.resize(face_frame, (256, 256)), 1.0, (256, 256),  (104.0, 177.0, 123.0)))

            for n in range(0, 68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

            # Calculate ratio of left and right eye
            roe_l = distance(landmarks.part(37).x, landmarks.part(37).y, landmarks.part(40).x, landmarks.part(40).y) / \
                    distance(landmarks.part(38).x, landmarks.part(38).y, landmarks.part(41).x, landmarks.part(41).y)
            roe_r = distance(landmarks.part(44).x, landmarks.part(44).y, landmarks.part(47).x, landmarks.part(47).y) / \
                    distance(landmarks.part(42).x, landmarks.part(42).y, landmarks.part(45).x, landmarks.part(45).y)
            roe_list.append(roe_l)
            roe_list.append(roe_r)
            roe = np.array(roe_list[-20:], dtype=np.float32)

            relative_face_width = (x2 - x1) / frame_width

            if relative_face_width > 0.35:  # checks whether person is too far away from camera or not
                if len(roe_list) >= 20:  # minimum 20 roe samples are needed for Eye Liveness network's prediction
                    is_Fake, confidence = spoof_detector.predict(blob, roe)

                    if confidence > 0.9:  # setting confidence threshold
                        if is_Fake:
                            label = "{}   {:0.3f}".format('Fake', confidence)
                            cv2.putText(frame, label, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        else:
                            label = "{}   {:0.3f}".format('Real', confidence)
                            cv2.putText(frame, label, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'Analyzing', (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 154, 252), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 154, 252), 2)

                else:
                    cv2.putText(frame, 'Analyzing', (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 154, 252), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 154, 252), 2)
            else:
                cv2.putText(frame, 'Too Far Away', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        frame_count += 1
        key = cv2.waitKey(1)
        if key == 27:
            # Plot Histogram of eye width/height ratio
            plt.hist(roe_list, bins=20)
            plt.title("Histogram of Eye width/height ratio")
            # plt.show()

            # Show runtime and FPS
            toc = time.perf_counter()
            print(f"Time: {toc - tic:0.2f} seconds, FPS: {frame_count / (toc - tic):0.2f}")

            # Write errors to csv
            for error in error_list:
                csv_writer.writerow([error])

            break  # press Esc to stop program

    except Exception as e:
        error_list.append(e)
