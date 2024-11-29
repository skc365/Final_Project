import cv2
import mediapipe as mp
import numpy as np
from time import time
import math
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp.pose.Pose(static_image_mode = True, min_dectection_confidence=0.3, model_complexity=2)

url = 'http://192.168.0.4:8080/video'

def detectPose(image, pose):
    
    # BGR -> RGB로 변환
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # 랜드마크 저장 리스트
    landmarks = []
    
    # 랜드마크가 감지되면,
    if results.pose_landmarks:
    
        # 랜드마크와 연결선 그리기
        mp_drawing.draw_landmarks(image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # 랜드마크 저장
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

        def calculateAngle(landmark1, landmark2, landmark3):

            #landmark 1,2,3 정의
            x1, y1, _ = landmark1
            x2, y2, _ = landmark2
            x3, y3, _ = landmark3

            #각 계산
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

            #각이 음수이면, 360도 더해주기
            if angle < 0:
                angle += 360

            return angle

        def classifyPose(landmarks, output_image, display=False):

            label = 'Unknown Pose'

            color = (0, 0, 255)  # Red

            left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

            right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

            left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

            right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

            left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

            right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])  #36:28
            #both arms are straight
            if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
                #shoulders at required angle
                if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle >80 and right_shoulder_angle < 110:

                    #Warrior II pose
                    #one leg is straight
                    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                        #the other leg is bended at required angle
                        if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                            label = 'Warrior II Pose'

                    #T pose
                    #check both legs are straight
                    if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

                        label = 'T pose'

            #tree pose
            #one leg is straight
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                #the other leg is bended at required angle
                if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

                    label = 'Tree Pose'

            if label != 'Unknown Pose':

                color = (0, 255, 0)  #green

            cv2.putText(output_image, label, (10,30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            if display:
                plt.figure(figsize=[10,10])
                plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
                
            else:
                return output_image, label

        image = cv2.imread('media/warriorIIpose.jpg')
        output_image, landmarks = detectPose(image, pose, display=False)
        if landmarks:
            classifyPose(landmarks, output_image, display=True)

        image = cv2.imread('media/Tpose.jpg')
        output_image, landmarks = detectPose(image, pose, display=False)
        if landmarks:
            classifyPose(landmarks, output_image, display=True)

        image = cv2.imread('Tpose1.jpg')
        output_image, landmarks = detectPose(image, pose, display=False)
        if landmarks:
            classifyPose(landmarks, output_image, display=True)
            
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
