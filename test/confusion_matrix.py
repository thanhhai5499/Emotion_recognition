import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

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

# Dự đoán nhãn cho tập kiểm tra
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y, axis=1)

# Tạo ma trận nhầm lẫn
cm = confusion_matrix(y_true_classes, y_pred_classes, normalize='true')

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Tăng kích thước phông chữ
ax = sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues', cbar_kws={"shrink": .8}, annot_kws={"size": 10})

# Đặt tiêu đề và nhãn trục
plt.ylabel('Actual labels', fontsize=14)
plt.xlabel('Predicted labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)

# Đặt giới hạn cho cột màu
cbar = ax.collections[0].colorbar
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_ticklabels(['0', '0.20', '0.40', '0.60', '0.80', '1.0'])

plt.show()
