import os
import math
import cv2
import joblib
import pandas as pd
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_landmarks(image, pose_model):

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose_model.process(imageRGB)
    
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None

def calculateAngle(p1, p2, p3):
    
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3
    
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360
        
    return angle

#포즈 랜드마크에서 각도 계산
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

#이미지와 포즈 라벨을 기반으로 데이터셋 생성
def generate_dataset(image_folder, pose_model):

    data = []

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)

        # 이미지 이름에서 라벨 가져오기 (확장자 제거)
        label = os.path.splitext(img_name)[0]
        
        # 이미지 읽기
        image = cv2.imread(img_path)
        if image is None:
            print(f"image error : {img_path}")
            continue

        # 랜드마크 추출
        landmarks = extract_landmarks(image, pose_model)
        if not landmarks:
            print(f"landmark error : {img_path}")
            continue

        # 데이터 추가
        features = calculate_pose_features(landmarks)
        features['label'] = label
        data.append(features)

    return pd.DataFrame(data)

#이미지 경로 및 라벨
image_folder = 'image' 

# 데이터셋 생성
df = generate_dataset(image_folder, pose_model)

# 데이터 분리
X = df.drop('label', axis=1)  # 각도 데이터
y = df['label']  # 라벨

# 모델 학습
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# 모델 저장
df.to_csv('pose_training_data.csv', index=False)

joblib.dump(clf, 'pose_classification.pkl')

print("Complete")
