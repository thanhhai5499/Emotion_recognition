# model.py

from tensorflow.keras.models import load_model
import os
import cv2
from keras_facenet import FaceNet
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

def load_trained_model(model_path='models/emotion_model.h5'):
    return load_model(model_path)

def train_face_recognition_model(user_data_dir):
    face_recognition_model = FaceNet()
    face_encodings = []
    user_labels = []

    for user_dir in os.listdir(user_data_dir):
        user_path = os.path.join(user_data_dir, user_dir)
        if os.path.isdir(user_path):
            for img_name in os.listdir(user_path):
                img_path = os.path.join(user_path, img_name)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = face_recognition_model.extract(img_rgb, threshold=0.95)

                if len(faces) > 0:
                    face_encoding = faces[0]['embedding']
                    face_encodings.append(face_encoding)
                    user_labels.append(user_dir)

    face_encodings = np.array(face_encodings)
    user_labels = np.array(user_labels)

    label_encoder = LabelEncoder()
    user_labels_encoded = label_encoder.fit_transform(user_labels)

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(face_encodings, user_labels_encoded)

    joblib.dump(classifier, 'data/face_recognition_model.pkl')
    joblib.dump(label_encoder, 'data/label_encoder.pkl')
