import cv2
import dlib
import os

def crop_face_from_image(root_path, output_root_path):
    # dlib의 얼굴 탐지기 생성
    detector = dlib.get_frontal_face_detector()

    for folder_name in os.listdir(root_path):
        # 입력 경로의 모든 파일 처리
        image_path = os.path.join(root_path, folder_name)
        for file_name in os.listdir(image_path):
            file_path = os.path.join(image_path, file_name)
            
            # 이미지 파일만 처리
            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # 이미지 읽기
            image = cv2.imread(file_path)
            if image is None:
                print(f"Cannot read {file_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 얼굴 탐지
            faces = detector(gray)

            if len(faces) == 0:
                print(f"No faces detected in {file_path}")
                continue

            for i, face in enumerate(faces):
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())

                # 얼굴 부분 크롭
                cropped_face = image[y:y+h, x:x+w]

                # 크롭한 얼굴 저장
                output_path = os.path.join(output_root_path, folder_name)
                os.makedirs(output_path, exist_ok=True)
                face_output_path = os.path.join(output_path, f"{os.path.splitext(file_name)[0]}_face_{i}.jpg")
                cv2.imwrite(face_output_path, cropped_face)

                print(f"Cropped face saved at {face_output_path}")

# 사용 예시
image_path = "DATA/data/test_ori"
output_path = "DATA/data/test"

if not os.path.exists(output_path):
    os.makedirs(output_path)

crop_face_from_image(image_path, output_path)
