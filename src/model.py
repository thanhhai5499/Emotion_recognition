import tensorflow as tf

def load_trained_model(path='models/emotion_model.h5'):
    return tf.keras.models.load_model(path)
