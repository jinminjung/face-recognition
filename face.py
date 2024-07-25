import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import time

# MediaPipe 얼굴 검출 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 얼굴 검출 객체 생성
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# 얼굴 인식 데이터 준비
known_face_encodings = []
known_face_names = []

# 라벨링을 위한 사람 이름과 이미지 파일 딕셔너리
people_dict = {
    "jinmin": "./images/jinmin.jpg",  # Ensure correct path
    "obama": "./images/obama.jpg"     # Ensure correct path
}

# 미리 등록된 얼굴 이미지를 읽고 인코딩
for name, img_file in people_dict.items():
    try:
        image = face_recognition.load_image_file(img_file)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
    except Exception as e:
        print(f"Error loading {img_file}: {e}")

# 웹캠 시작
cap = cv2.VideoCapture(0)
prevTime = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curTime = time.time()  # current time
    fps = 1 / (curTime - prevTime)
    prevTime = curTime
    # 프레임 수 문자열에 저장
    fps_str = "FPS : %0.1f" % fps

    cv2.putText(frame, fps_str, (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

    # BGR 이미지를 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_detection.process(rgb_frame)

    if result.detections:
        for detection in result.detections:
            # 검출된 얼굴의 경계 상자 좌표 추출
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # 얼굴 영역 추출
            face_region = frame[y:y + h, x:x + w]
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

            # 얼굴 인코딩
            face_encodings = face_recognition.face_encodings(rgb_face)
            name = "Unknown"
            
            if face_encodings:
                face_encoding = face_encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # 얼굴에 사각형 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Detection and Labeling", frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
