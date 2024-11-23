# Final_Project 진행 상황

## 2024/11/08 1차

### 브레인 스토밍을 통한 주제 선정

1. 여러가지 손짓을 이용하여 플레이하는 러닝액션 게임   
	-> 과거 프로젝트과 차별점이 많아보이지 않음. -> 보류

2. 특정 SNS(ex YouTube, Instagram)를 크롤링해서 재가공하는 것   
	-> 재가공한 후 활용하는 부분에서 멈춤 -> 보류

3. 여러가지 몸짓을 이용하여 플레이하는 게임 ( 예 : https://www.youtube.com/watch?v=LKcbQHVB3xM )   
	-> 과거 프로젝트와 차별점이 충분히 있어보이고, 재미있어 보임 -> **선정**   
	-> 이미지 인식 및 처리는 MediaPipe를 기준으로 두고 추가로 공부하면서 논의   
	( 참고 : https://chuoling.github.io/mediapipe/solutions/pose.html )   

### 프로젝트를 위한 깃허브 

1. 깃허브 리포지토리 생성 ( https://github.com/skc365/Final_Project.git )   

2. 1차 진행(브레인스토밍) 내용 정리 및 커밋

## 2024/11/16 2차

### 자세 인식 코드
자세 인식을 할 수 있는 코드를 완성함. (참고: https://youtu.be/06TE_U21FK4?si=00_L4RfZhQEjxaQ2)

### 게임 내용 구체화
1. 컴퓨터가 랜덤으로 자세를 제시함
2. 일정 시간 안에 제시한 자세와 일치하면 점수를 얻음

### 파트 분담
- 자세가 일치하면 점수를 추가하는 파트 => score branch
- 각도를 계산하여 자세를 구현하는 파트 => pose branch (참고: https://www.youtube.com/watch?v=aySurynUNAw)

## 2024/11/23 3차

### score branch
- 일정 시간 내에 제시한 자세가 일치하면 점수를 얻는 부분 **해결됨**

### pose branch
- 포즈 이미지 파일의 각도와 내가 취하는 자세의 각도를 계산하고 일치하면 포즈 이미지와 같음을 알려주는 부분 **4차때까지 보완**
- 더 많은 포즈 이미지를 구해야 함 (참고 사이트: pixabay, unsplash, and etc.)
