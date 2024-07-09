import time
import cv2
import dlib
import numpy as np  # Thêm dòng này
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

    def process_frame(self, frame):
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

    def predict_emotion(self, image):
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None
        predictions = self.model.predict(processed_image)
        emotion = np.argmax(predictions)
        return emotion

    def get_emotion_text(self, emotion):
        emotions = ["Tức giận", "Ghê tởm", "Sợ hãi", "Vui vẻ", "Trung tính", "Buồn", "Ngạc nhiên"]
        if emotion < len(emotions):
            return emotions[emotion]
        return "Không xác định"
