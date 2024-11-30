import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import random

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

url = 'http://192.168.0.4:8080/video'

# 초기 점수
score = 0

# 포즈 감지 함수
def detectPose(image, pose):
    
    # BGR -> RGB로 변환
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(imageRGB)
    
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
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))

    return image, landmarks

# 랜덤 이미지 파일 이름 반환
def get_random_image(folder_path):
    
    #폴더 내에서 이미지 파일 리스트 가져오기
    images = os.listdir(folder_path)
    
    #랜덤으로 이미지 선택
    random_image = random.choice(images)

    return os.path.join(folder_path, random_image), os.path.splitext(random_image)[0]

# 세 랜드마크 사이의 각도 계산 함수
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

# 포즈 분류 함수
def classifyPose(landmarks):

    #기본 라벨은 Unknown으로 지정
    label = 'Unknown'

    if not landmarks or len(landmarks) < 33:  # 랜드마크가 적은 경우,
        return label

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
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #두 팔이 펴지고,
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        #어깨가 특정 각도에 있고,
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle >80 and right_shoulder_angle < 110:

            #Warrior pose
            #한 다리가 직선이고,
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                #다른 다리가 특정 각도만큼 굽혀져 있을 때,
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                    label = 'warrior'
    '''
    각도 추가 해야 함
    '''
    return label

# 비디오 처리 및 포즈 평가
def process_video(url, pose_model, target_label, target_image):
    
    global score

    #target_image 크기 조정
    resize_width = 320
    target_height, target_width, _ = target_image.shape
    resize_height = int((resize_width / target_width) * target_height)
    resized_image = cv2.resize(target_image, (resize_width, resize_height))

    #조정된 이미지 확인 디버깅
    print(f"Target image resized to: {resized_image.shape}")

    video = cv2.VideoCapture(url)
    video.set(3, 1280)  # 비디오 너비 설정
    video.set(4, 960)  # 비디오 높이 설정

    # 타이머 시작
    start_time = time.time()

    # 10초 타이머
    while video.isOpened() and (time.time() - start_time < 10):
        success, frame = video.read()
        if not success:
            break

        #영상 좌우 반전
        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        # 전체 창 크기 설정
        combined_width = resize_width + frame.shape[1]
        combined_height = max(frame.shape[0], resize_height)
    
        # 랜드마크 연결선을 먼저 그리기 위해
        frame, landmarks = detectPose(frame, pose_model)

        # 빈 공간 생성 (전체 화면 크기)
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        # 이미지 추가 (왼쪽)
        combined_frame[:resize_height, :resize_width] = resized_image

        # 경계선 추가
        cv2.line(combined_frame, (resize_width, 0), (resize_width, combined_height), (255, 255, 255), 2)

        # 동영상 프레임 추가 (오른쪽)
        combined_frame[:frame.shape[0], resize_width:] = frame

        # 포즈의 랜드마크 감지
        user_pose_label = classifyPose(landmarks)

        # 남은 시간
        remaining_time = max(0, 10 - int(time.time() - start_time))

        # 점수 및 상태 텍스트 추가 (왼쪽 하단 빈 공간에 추가)
        text_x = 10
        text_y = resize_height + 30
        
        # 점수
        if user_pose_label == target_label:
            score += 10  # 점수 증가
            cv2.putText(combined_frame, "Matched! +10", 
                        (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break  # 포즈가 맞으면 다음 이미지
        else:
            cv2.putText(combined_frame, "Not Matched", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(combined_frame, f"Target Pose: {target_label}", (text_x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(combined_frame, f"Your Pose: {user_pose_label}", (text_x, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(combined_frame, f"Score: {score}", (text_x, text_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(combined_frame, f"Time Left: {remaining_time}s", (text_x, text_y + 120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.namedWindow("Pose Match", cv2.WINDOW_NORMAL)
        cv2.imshow("Pose Match", combined_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            return "exit"
        elif key == ord('e'):
            time.sleep(1)
            video.release()
            cv2.destroyAllWindows()
            return "continue"

    video.release()
    cv2.destroyAllWindows()

def main():
    image_folder = 'image'

    #이미지 폴더 디버깅
    if not os.path.exists(image_folder):
        print(f"Image folder error : {image_folder}")
        return
    
    while True:
        random_image_path, target_label = get_random_image(image_folder)
        
        target_image = cv2.imread(random_image_path)
        
        # 랜덤 이미지 디버깅
        if target_image is None:
            print(f"Target image error : {random_image_path}")
            continue

        pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

        result = process_video(url, pose_video, target_label, target_image)

        if result == "exit":
            print("Exiting program.")
            break

if __name__ == "__main__":
    main()
