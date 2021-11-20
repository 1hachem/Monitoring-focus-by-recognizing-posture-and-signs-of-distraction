import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import time
#import statsmodels.api as sm
import csv
import subprocess
import os
from threading import Thread


def launch_read():
    os.system("start pythonw read.pyw")

thread = Thread(target = launch_read)
    

launched_read = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands_module = mp.solutions.hands


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

positions = defaultdict(list)
face_edge_r = defaultdict(list)
face_edge_l = defaultdict(list)
#subprocess.call(['python', 'read.py'])
data = []
elapsed_time = []
with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=["state","yawn","closed","hands_on_face","away","time"])
    csv_writer.writeheader()

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    with mp_hands_module.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
        start = time.time()
        frames = 0
        n_points = 21

        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            font = cv2.FONT_HERSHEY_SIMPLEX
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(image)
            results_hands = hands.process(image)
            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            state = 1
            yawn = 0
            hands_on_face = 0
            closed = 0
            away = 0

            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    face = []
                    for id, lm in enumerate(face_landmarks.landmark):
                        ih, iw, ic = image.shape
                        x, y, z = lm.x, lm.y, lm.z
                        face.append([x, y, z])

                    ext = [234, 454, 10, 152]

                    a = np.sqrt((face[234][0]-face[454][0]) **
                                2+(face[234][1]-face[454][1])**2)

                    b = np.sqrt((face[10][0]-face[152][0]) **
                                2+(face[10][1]-face[152][1])**2)

                   
                    thresh1 = 0.21
                    thresh2 = 0.4
            

                    if a < thresh1:
                        state=0
                        cv2.putText(image, "distracted", (int(frameWidth / 2.5), 50), font,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                        away = 1

                    if b < thresh2:
                        state=0
                        cv2.putText(image, "distracted", (int(frameWidth / 2.5), 50), font,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                        away = 1


                    d5 = np.sqrt((face[160][0] - face[144][0])**2 + (face[160][1] - face[144][1])**2)
                    d6 = np.sqrt((face[158][0] - face[153][0])**2 + (face[158][1] - face[153][1])**2)
                    d7 = np.sqrt((face[33][0] - face[133][0])**2 + (face[33][1] - face[133][1])**2)

                    d8 = np.sqrt((face[385][0] - face[380][0])**2 + (face[385][1] - face[380][1])**2)
                    d9 = np.sqrt((face[387][0] - face[373][0])**2 + (face[387][1] - face[373][1])**2)
                    d10 = np.sqrt((face[362][0] - face[263][0])**2 + (face[362][1] - face[263][1])**2)

                    thresh_eye = 0.25

                    if (d5 + d6) / (2 * d7) < thresh_eye and (d8 + d9) / (2 * d10) < thresh_eye:
                        cv2.putText(image, "distracted", (int(frameWidth / 2.5), 50), font,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                        closed = 1

                    d11 = np.sqrt((face[13][0] - face[14][0])**2 + (face[13][1] - face[14][1])**2)
                    d12 = np.sqrt((face[78][0] - face[308][0])**2 + (face[78][1] - face[308][1])**2)

                    thresh_yawn = 1

                    if d11 / d12 > thresh_yawn:
                        state=0
                        cv2.putText(image, "distracted", (int(frameWidth / 2.5), 50), font,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                        yawn = 1


                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )
                    """mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )"""
                    """mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())"""
            else:
                state = 0
                cv2.putText(image, "distracted", (int(frameWidth / 2.5), 50), font,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
            # Flip the image horizontally for a selfie-view display.

            face_r_x_c = 0
            face_r_y_c = 0

            face_l_x_c = 0
            face_l_y_c = 0

            hand_1_x_c = 0
            hand_1_y_c = 0

            hand_2_x_c = 0
            hand_2_y_c = 0

            if results_face.multi_face_landmarks:
                r = [377, 400, 378, 379, 365, 397, 367, 435, 433,
                     401, 366, 447, 389, 372, 251, 284, 332, 297]
                l = [148, 176, 140, 149, 170, 169, 136, 172, 138, 215,
                     177, 137, 227, 143, 162, 21, 54, 103, 67, 109]
                for face in results_face.multi_face_landmarks:
                    i = 0
                    for landmark in face.landmark:
                        if i in r or i in l:
                            x = landmark.x
                            y = landmark.y

                            shape = image.shape
                            relative_x = int(x * shape[1])
                            relative_y = int(y * shape[0])

                            if i in r:
                                face_edge_r[i].append((relative_x, relative_y))
                                face_r_x_c += int(relative_x / len(r))
                                face_r_y_c += int(relative_y / len(r))

                            if i in l:
                                face_edge_l[i].append((relative_x, relative_y))
                                face_l_x_c += int(relative_x / len(l))
                                face_l_y_c += int(relative_y / len(l))

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.circle(image, (relative_x, relative_y),
                                       radius=5, color=(225, 0, 100), thickness=1)
                        i += 1

            if results_hands.multi_hand_landmarks != None:
                i = 0
                for hand in results_hands.multi_hand_landmarks:
                    for landmark in hand.landmark:
                        x = landmark.x
                        y = landmark.y

                        shape = image.shape
                        relative_x = int(x * shape[1])
                        relative_y = int(y * shape[0])

                        positions[i].append((relative_x, relative_y))

                        if i < 21:
                            hand_1_x_c += int(relative_x / n_points)
                            hand_1_y_c += int(relative_y / n_points)

                        else:
                            hand_2_x_c += int(relative_x / n_points)
                            hand_2_y_c += int(relative_y / n_points)

                    mp_drawing.draw_landmarks(
                        image, hand, mp_hands_module.HAND_CONNECTIONS)
                    H = hand.landmark
                    cv2.circle(image, (int(H[9].x*shape[1]), int(H[9].y*shape[0])),
                               radius=5, color=(225, 255, 255), thickness=5)
                    data.append(H[9].x)

                    i += 1


            d1 = np.sqrt((hand_1_x_c-face_l_x_c)**2+(hand_1_y_c-face_l_y_c)**2)
            d2 = np.sqrt((hand_1_x_c-face_r_x_c)**2+(hand_1_y_c-face_r_y_c)**2)
            d3 = np.sqrt((hand_2_x_c-face_l_x_c)**2+(hand_2_y_c-face_l_y_c)**2)
            d4 = np.sqrt((hand_2_x_c-face_r_x_c)**2+(hand_2_y_c-face_r_y_c)**2)
            d = [d1, d2, d3, d4]
            hands_on_face = 0
            thresh3 = 100

            if any([i < thresh3 for i in d]):
                state = 0
                cv2.putText(image, "distracted", (int(frameWidth / 2.5), 50), font,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                hands_on_face=1

            """cv2.putText(image, 'c', (face_r_x_c, face_r_y_c),
                        font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'c', (face_l_x_c, face_l_y_c),
                        font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'c', (hand_1_x_c, hand_1_y_c),
                        font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'c', (hand_2_x_c, hand_2_y_c),
                        font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)"""
           
            cv2.imshow('image', image)
            end = time.strftime("%H:%M:%S", time.localtime(time.time()))


            with open('data.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file,fieldnames=["state","yawn","closed","hands_on_face","away","time"])

                info = {
                    "state": state,
                    "yawn":yawn,
                    "closed":closed,
                    "hands_on_face":hands_on_face,
                    "away":away,
                    "time":end,

                }

                csv_writer.writerow(info)

            if (not launched_read) and (time.time() - start > 0.3):
                thread.start()
                launched_read = True
    
            # closing functionalities
            if cv2.waitKey(5) & 0xFF == 27:
                break
            if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
                break
cap.release()


