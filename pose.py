import os
import joblib
import cv2
import mediapipe as mp
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

url = 'http://192.168.0.4:8080/video'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# 학습된 모델 로드
model = joblib.load('pose_classification.pkl')

def detectPose(image, pose):
    
    output_image = image.copy()
    
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(imageRGB)
    
    height, width, _ = image.shape
    
    landmarks = []
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append(landmark)
            
    return output_image, landmarks

def calculateAngle(p1, p2, p3):
    
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360
        
    return angle

def calculate_pose_features(landmarks):
    
    features = {
        'left_elbow_angle': calculateAngle(
            (landmarks[11].x, landmarks[11].y, landmarks[11].z),
            (landmarks[13].x, landmarks[13].y, landmarks[13].z),
            (landmarks[15].x, landmarks[15].y, landmarks[15].z)
        ),
        'right_elbow_angle': calculateAngle(
            (landmarks[12].x, landmarks[12].y, landmarks[12].z),
            (landmarks[14].x, landmarks[14].y, landmarks[14].z),
            (landmarks[16].x, landmarks[16].y, landmarks[16].z)
        ),
        'left_shoulder_angle': calculateAngle(
            (landmarks[13].x, landmarks[13].y, landmarks[13].z),
            (landmarks[11].x, landmarks[11].y, landmarks[11].z),
            (landmarks[23].x, landmarks[23].y, landmarks[23].z)
        ),
        'right_shoulder_angle': calculateAngle(
            (landmarks[14].x, landmarks[14].y, landmarks[14].z),
            (landmarks[12].x, landmarks[12].y, landmarks[12].z),
            (landmarks[24].x, landmarks[24].y, landmarks[24].z)
        ),
        'left_knee_angle': calculateAngle(
            (landmarks[23].x, landmarks[23].y, landmarks[23].z),
            (landmarks[25].x, landmarks[25].y, landmarks[25].z),
            (landmarks[27].x, landmarks[27].y, landmarks[27].z)
        ),
        'right_knee_angle': calculateAngle(
            (landmarks[24].x, landmarks[24].y, landmarks[24].z),
            (landmarks[26].x, landmarks[26].y, landmarks[26].z),
            (landmarks[28].x, landmarks[28].y, landmarks[28].z)
        ),
        'left_ankle_angle': calculateAngle(
            (landmarks[25].x, landmarks[25].y, landmarks[25].z),  
            (landmarks[27].x, landmarks[27].y, landmarks[27].z),  
            (landmarks[31].x, landmarks[31].y, landmarks[31].z)   
        ),
        'right_ankle_angle': calculateAngle(
            (landmarks[26].x, landmarks[26].y, landmarks[26].z),  
            (landmarks[28].x, landmarks[28].y, landmarks[28].z),  
            (landmarks[32].x, landmarks[32].y, landmarks[32].z)   
        ),
        'left_hip_angle': calculateAngle(
            (landmarks[11].x, landmarks[11].y, landmarks[11].z),  
            (landmarks[23].x, landmarks[23].y, landmarks[23].z),  
            (landmarks[25].x, landmarks[25].y, landmarks[25].z)   
        ),
        'right_hip_angle': calculateAngle(
            (landmarks[12].x, landmarks[12].y, landmarks[12].z),  
            (landmarks[24].x, landmarks[24].y, landmarks[24].z),  
            (landmarks[26].x, landmarks[26].y, landmarks[26].z)   
        )
    }
    
    return features

def classifyPose(landmarks, output_image, model):

    color = (0, 255, 0)
    
    features = calculate_pose_features(landmarks)
    
    feature_names = [
        'left_elbow_angle', 'right_elbow_angle',
        'left_shoulder_angle', 'right_shoulder_angle',
        'left_knee_angle', 'right_knee_angle',
        'left_ankle_angle', 'right_ankle_angle',
        'left_hip_angle', 'right_hip_angle'
    ]
    
    features_values = pd.DataFrame([list(features.values())], columns=feature_names)

    # 예측 확률
    probabilities = model.predict_proba(features_values)[0]
    max_prob = max(probabilities)

    # 예측 클래스와 확률 확인
    predicted_pose = model.classes_[np.argmax(probabilities)]
    print(max_prob)
    # 확률이 0.6 미만이면 Unknown Pose로 처리
    if max_prob < 0.5:
        color = (0, 0, 255)
        predicted_pose = "Unknown Pose"
    
    cv2.putText(output_image, f"Pose: {predicted_pose}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return output_image, predicted_pose

def process_video(url, pose_model, model):
    
    video = cv2.VideoCapture(url)
    
    while video.isOpened():
        
        success, frame = video.read()
        
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        frame_height, frame_width, _ = frame.shape
        
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        frame, landmarks = detectPose(frame, pose_model)
        
        if landmarks:
            frame, _ = classifyPose(landmarks, frame, model)
            
        cv2.imshow("Pose Classification", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
    video.release()
    cv2.destroyAllWindows()

# 실행
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
process_video(url, pose_video, model)
