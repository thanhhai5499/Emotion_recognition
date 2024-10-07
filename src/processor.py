import time
import cv2
import dlib
import numpy as np
from src.preprocess import preprocess_image
from src.model import load_trained_model

class EmotionRecognitionProcessor:
    def __init__(self):
        self.model = load_trained_model()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
        self.last_update_time = time.time()
        self.emotion_start_time = None
        self.current_emotion = None

    def process_frame1(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        landmarks = []
        for face in faces:
            shape = self.predictor(gray, face)
            landmarks.append(shape)
            for n in range(0, 68):
                x = shape.part(n).x
                y = shape.part(n).y
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        return frame, faces, landmarks    

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        landmarks = []
        for face in faces:
            shape = self.predictor(gray, face)
            landmarks.append(shape)

            # Lấy ba điểm 28, 29, 30
            p28 = (shape.part(28).x, shape.part(28).y)
            p29 = (shape.part(29).x, shape.part(29).y)
            p30 = (shape.part(30).x, shape.part(30).y)

            # Tính góc của trục tung
            dx = p29[0] - p28[0]
            dy = p29[1] - p28[1]
            angle = np.arctan2(dy, dx) * 180.0 / np.pi

            # Xác định hình vuông bao quanh khuôn mặt, to hơn một chút so với khuôn mặt
            square_size = max(face.width(), face.height()) * 1.5
            center_x, center_y = p30
            x_min = int(center_x - square_size / 2)
            x_max = int(center_x + square_size / 2)
            y_min = int(center_y - square_size / 2)
            y_max = int(center_y + square_size / 2)

            # Vẽ hình vuông bao quanh khuôn mặt
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Vẽ trục OX và OY trong hình vuông, với điểm 30 làm tâm
            length = square_size / 2
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))

            pX1 = (int(center_x + length * cos_a), int(center_y + length * sin_a))
            pX2 = (int(center_x - length * cos_a), int(center_y - length * sin_a))
            pY1 = (int(center_x - length * sin_a), int(center_y + length * cos_a))
            pY2 = (int(center_x + length * sin_a), int(center_y - length * cos_a))

            cv2.line(frame, pX1, pX2, (0, 255, 0), 2)
            cv2.line(frame, pY1, pY2, (0, 255, 0), 2)

            # Vẽ các đường lưới màu xám mờ song song với OX và OY
            step_size = 20  # Khoảng cách giữa các đường lưới
            for i in range(1, int(square_size // (2 * step_size)) + 1):
                # Đường song song với OX
                offset = i * step_size
                cv2.line(frame, 
                         (int(center_x + offset * cos_a), int(center_y + offset * sin_a)), 
                         (int(center_x + offset * cos_a - square_size * sin_a / length), 
                          int(center_y + offset * sin_a + square_size * cos_a / length)), 
                         (200, 200, 200), 1)

                cv2.line(frame, 
                         (int(center_x - offset * cos_a), int(center_y - offset * sin_a)), 
                         (int(center_x - offset * cos_a - square_size * sin_a / length), 
                          int(center_y - offset * sin_a + square_size * cos_a / length)), 
                         (200, 200, 200), 1)

                # Đường song song với OY
                cv2.line(frame, 
                         (int(center_x + offset * sin_a), int(center_y - offset * cos_a)), 
                         (int(center_x + offset * sin_a - square_size * cos_a / length), 
                          int(center_y - offset * cos_a - square_size * sin_a / length)), 
                         (200, 200, 200), 1)

                cv2.line(frame, 
                         (int(center_x - offset * sin_a), int(center_y + offset * cos_a)), 
                         (int(center_x - offset * sin_a - square_size * cos_a / length), 
                          int(center_y + offset * cos_a - square_size * sin_a / length)), 
                         (200, 200, 200), 1)

        return frame, faces, landmarks

    def predict_emotion(self, image):
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None
        predictions = self.model.predict(processed_image)
        emotion = np.argmax(predictions)
        return emotion

    def get_emotion_text(self, emotion):
        abnormal_emotions = [0, 1, 2, 4, 5]
        normal_emotions = [3, 6]

        if emotion in abnormal_emotions:
            return "Bất thường"
        elif emotion in normal_emotions:
            return "Bình thường"
        return "Không xác định"
