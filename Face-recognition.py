import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import time
from collections import deque

# 얼굴 인식의 기준이 될 이미지 로드 및 인코딩 리스트
known_face_encodings = []
known_face_names = []

# 예를 들어, 'person1', 'person2' 등으로 인물 구분
face_image_dirs = {
    'Person1/jinmin': ['images/jinmin.jpg'],
    'Person2/obama' :['images/obama.jpg']
}

for name, image_paths in face_image_dirs.items():
    encodings = []
    for image_path in image_paths:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            encodings.append(face_encodings[0])
    
    if encodings:
        mean_encoding = np.mean(encodings, axis=0)
        known_face_encodings.append(mean_encoding)
        known_face_names.append(name)

# 영상 장치 초기에 초기화
cap = cv2.VideoCapture(0)
pTime = 0
fps_list = deque(maxlen=30)  # 최근 30개의 FPS 값을 저장할 덱

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

TOLERANCE = 0.6

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.2) as face_detection:
    frame_skip = 5  # 매 5번째 프레임마다 얼굴 인식 수행
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        if frame_count % frame_skip == 0:
            results = face_detection.process(frame_rgb)
            face_locations = []
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame_rgb.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    top_left = (bbox[0], bbox[1])
                    bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                    face_locations.append((top_left[1], bottom_right[0], bottom_right[1], top_left[0]))

            face_encodings = face_recognition.face_encodings(frame_rgb, face_locations, num_jitters=1)
            face_names = []
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if face_distances.size > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                
                face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            scale_factor = 2  # 이전에 이미지를 0.5로 줄였으므로 되돌리기 위해 2를 곱함
            cv2.rectangle(frame, (left * scale_factor, top * scale_factor), (right * scale_factor, bottom * scale_factor), (0, 255, 0), 2)
            cv2.rectangle(frame, (left * scale_factor, (bottom - 35) * scale_factor), (right * scale_factor, bottom * scale_factor), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left * scale_factor + 6, (bottom - 6) * scale_factor), font, 1.0, (255, 255, 255), 1)
        
        # fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        fps_list.append(fps)
        avg_fps = sum(fps_list) / len(fps_list)
        cv2.putText(frame, f'FPS: {int(avg_fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
