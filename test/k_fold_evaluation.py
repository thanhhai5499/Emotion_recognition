import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
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

# K-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
correct_predictions = []
incorrect_predictions = []
accuracies = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    correct = np.sum(y_pred_classes == y_true_classes)
    incorrect = len(y_pred_classes) - correct
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    
    correct_predictions.append(correct)
    incorrect_predictions.append(incorrect)
    accuracies.append(accuracy * 100)

# Tạo bảng kết quả
results = {
    'n-Fold': [f'{i+1}-fold' for i in range(k)],
    'Correct': correct_predictions,
    'Errors': incorrect_predictions,
    'Accuracy': [f'{acc:.2f}%' for acc in accuracies]
}

# Thêm dòng tổng
results['n-Fold'].append('Overall')
results['Correct'].append(np.sum(correct_predictions))
results['Errors'].append(np.sum(incorrect_predictions))
results['Accuracy'].append(f'{np.mean(accuracies):.2f}%')

# In bảng theo định dạng mong muốn
print(tabulate(results, headers="keys", tablefmt="grid"))
