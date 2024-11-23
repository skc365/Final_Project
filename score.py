import cv2
import mediapipe as mp
import numpy as np
import time
import random


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


GESTURES = ['Next level', 'Supernova']
score = 0


#세 점의 각도를 계산하는 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_random_gesture():
    return random.choice(GESTURES)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    gesture_to_match = get_random_gesture()
    gesture_text = 'Cant found gesture'
    start_time = time.time()
    time_limit = 10
    
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark


            #왼쪽 어깨
            left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            #왼쪽 팔꿈치
            left_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            #왼쪽 손목
            left_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            #왼쪽 팔꿈치 각도
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            if 60 < angle < 120:
                gesture_text = 'Next level'
            if angle > 150:
                gesture_text = 'Supernova'

            if gesture_text == gesture_to_match:
                elapsed_time = time.time() - start_time
                if elapsed_time <= time_limit:
                    score += 10
                    gesture_to_match = get_random_gesture()
                    start_time = time.time()

        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0:
            gesture_to_match = get_random_gesture()
            start_time = time.time()

        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66)),
                                  mp_drawing.DrawingSpec(color=(245,66,230)))

        cv2.putText(image, f"Match Gesture: {gesture_to_match}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f'Your Gesture: {gesture_text}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(image, f'Score: {score}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)
        cv2.putText(image, f"Time Left: {int(remaining_time)}s", (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
