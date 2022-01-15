import cv2
import dlib
import numpy as np
import time
import csv

cap = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")

LABEL = 0

def distance(x1, y1, x2, y2):
    # Returns euclidean distance between two points
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

roi_list = []
frame_count = 0
n_samples = 0
tic = time.perf_counter()

with open('eye_ratio_f.csv', mode='w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    while True:
        try:
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                roi_l = distance(landmarks.part(37).x, landmarks.part(37).y, landmarks.part(40).x, landmarks.part(40).y) / \
                      distance(landmarks.part(38).x, landmarks.part(38).y, landmarks.part(41).x, landmarks.part(41).y)
                roi_r = distance(landmarks.part(44).x, landmarks.part(44).y, landmarks.part(47).x, landmarks.part(47).y) / \
                        distance(landmarks.part(42).x, landmarks.part(42).y, landmarks.part(45).x, landmarks.part(45).y)
                roi_list.append(roi_l)
                roi_list.append(roi_r)
                rl = np.array(roi_list[-20:], dtype=np.float32)

                print(frame_count-8, '/100')
                if frame_count>8:
                    n_samples += 1
                    csv_writer.writerow([*roi_list[-20:], LABEL])

                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

            cv2.imshow("Frame", frame)
            frame_count += 1
            key = cv2.waitKey(1)
            if key == 27 or frame_count-1 == 108:
                break 

        except:
            print("Frame skipped")
