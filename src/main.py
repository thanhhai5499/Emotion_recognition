import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from src.camera import RealSenseCamera
from src.preprocess import preprocess_image
from src.model import load_trained_model
import numpy as np

def predict_emotion(image, model):
    processed_image = preprocess_image(image)
    if processed_image is None:
        return None
    predictions = model.predict(processed_image)
    emotion = np.argmax(predictions)
    return emotion

def main():
    camera = RealSenseCamera()
    model = load_trained_model()

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            emotion = predict_emotion(frame, model)
            if emotion is not None:
                print(f'Predicted emotion: {emotion}')
            else:
                print("No emotion detected")

            cv2.imshow('RealSense', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
