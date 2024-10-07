import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tabulate import tabulate

# Load dữ liệu
def load_fer2013(data_path):
    data = pd.read_csv(data_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = face / 255.0
        faces.append(face)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).to_numpy()
    return np.array(faces), emotions

X, y = load_fer2013('data/fer2013.csv')

# Load mô hình đã huấn luyện
model = load_model('models/emotion_model.h5')

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Dự đoán
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Tạo confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Tính toán accuracy cho từng class
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Tính toán overall accuracy
overall_accuracy = accuracy_score(y_true_classes, y_pred_classes)

# Tính toán average accuracy
average_accuracy = class_accuracies.mean()

# Tạo bảng kết quả với chỉ hai dòng: "Test" và "Average"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
results = {
    'Test': [f'{acc * 100:.2f}' for acc in class_accuracies] + [f'{overall_accuracy * 100:.2f}'],
    'Average': [f'{average_accuracy * 100:.2f}'] * len(emotion_labels) + [f'{average_accuracy * 100:.2f}']
}

# Chuyển đổi định dạng bảng
df = pd.DataFrame(results, index=emotion_labels + ['Overall']).T

# In bảng kết quả
print(tabulate(df, headers="keys", tablefmt="grid"))



