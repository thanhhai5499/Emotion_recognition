import sys
import os
import cv2
import time
import joblib
import numpy as np
import pandas as pd  # Thêm pandas để in bảng
from sklearn.metrics import classification_report, accuracy_score

# Thêm đường dẫn của thư mục src vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from camera import RealSenseCameraNew

def evaluate_models(camera, knn_model, svm_model, input_label, labels):
    print(f"Đang quét nhãn: {input_label}")
    knn_predictions = []
    svm_predictions = []
    true_labels = []

    start_time = time.time()
    while time.time() - start_time < 20:  # Quét trong 20 giây
        ret, color_frame, depth_frame = camera.get_frames()  # Nhận cả color_frame và depth_frame
        if not ret:
            print(f"Không thể lấy khung hình.")
            continue

        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        flattened_frame = gray_frame.flatten().reshape(1, -1)

        knn_prediction = knn_model.predict(flattened_frame)
        svm_prediction = svm_model.predict(flattened_frame)

        knn_predictions.append(knn_prediction[0])
        svm_predictions.append(svm_prediction[0])
        true_labels.append(input_label)

        # Hiển thị hình ảnh với OpenCV
        cv2.imshow("RealSense Camera - Quét Nhãn", color_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Đánh giá mô hình KNN
    knn_report = classification_report(true_labels, knn_predictions, output_dict=True, labels=labels)
    accuracy_knn = accuracy_score(true_labels, knn_predictions) * 100

    print(f"\nĐộ chính xác KNN: {accuracy_knn:.2f}%")

    # Đánh giá mô hình SVM
    svm_report = classification_report(true_labels, svm_predictions, output_dict=True, labels=labels)
    accuracy_svm = accuracy_score(true_labels, svm_predictions) * 100

    print(f"\nĐộ chính xác SVM: {accuracy_svm:.2f}%")

    # So sánh với tất cả các nhãn
    knn_results = []
    svm_results = []

    # Lưu trữ tổng kết phần trăm để tính trung bình
    knn_total_precision, knn_total_recall, knn_total_f1 = 0, 0, 0
    svm_total_precision, svm_total_recall, svm_total_f1 = 0, 0, 0

    count_labels = len(labels)

    for label, metrics in knn_report.items():
        if isinstance(metrics, dict):
            precision = metrics['precision'] * 100
            recall = metrics['recall'] * 100
            f1_score = metrics['f1-score'] * 100
            knn_results.append([label, f"{precision:.2f}%", f"{recall:.2f}%", f"{f1_score:.2f}%"])

            # Cộng dồn để tính trung bình
            knn_total_precision += precision
            knn_total_recall += recall
            knn_total_f1 += f1_score

    for label, metrics in svm_report.items():
        if isinstance(metrics, dict):
            precision = metrics['precision'] * 100
            recall = metrics['recall'] * 100
            f1_score = metrics['f1-score'] * 100
            svm_results.append([label, f"{precision:.2f}%", f"{recall:.2f}%", f"{f1_score:.2f}%"])

            # Cộng dồn để tính trung bình
            svm_total_precision += precision
            svm_total_recall += recall
            svm_total_f1 += f1_score

    # Tính trung bình
    knn_avg_precision = knn_total_precision / count_labels
    knn_avg_recall = knn_total_recall / count_labels
    knn_avg_f1 = knn_total_f1 / count_labels

    svm_avg_precision = svm_total_precision / count_labels
    svm_avg_recall = svm_total_recall / count_labels
    svm_avg_f1 = svm_total_f1 / count_labels

    # Chuyển đổi thành DataFrame để in bảng
    knn_df = pd.DataFrame(knn_results, columns=["Nhãn", "Precision", "Recall", "F1-Score"])
    svm_df = pd.DataFrame(svm_results, columns=["Nhãn", "Precision", "Recall", "F1-Score"])

    print("\nBảng so sánh KNN:")
    print(knn_df.to_string(index=False))

    print("\nBảng so sánh SVM:")
    print(svm_df.to_string(index=False))

    # In trung bình phần trăm
    print(f"\nTrung bình KNN - Precision: {knn_avg_precision:.2f}%, Recall: {knn_avg_recall:.2f}%, F1-Score: {knn_avg_f1:.2f}%")
    print(f"Trung bình SVM - Precision: {svm_avg_precision:.2f}%, Recall: {svm_avg_recall:.2f}%, F1-Score: {svm_avg_f1:.2f}%")

    # Tổng trung bình phần trăm nhận diện
    total_avg = (knn_avg_precision + svm_avg_precision) / 2
    print(f"\nTổng trung bình phần trăm nhận diện khuôn mặt: {total_avg:.2f}%")

if __name__ == "__main__":
    # Đường dẫn mô hình đã lưu
    knn_model_path = os.path.join('models', 'knn_model.pkl')
    svm_model_path = os.path.join('models', 'svm_model.pkl')

    # Load mô hình KNN và SVM
    knn_model = joblib.load(knn_model_path)
    svm_model = joblib.load(svm_model_path)

    # Nhập nhãn mà bạn muốn so sánh
    input_label = input("Nhập nhãn để kiểm tra (Tên người dùng): ")

    # Lấy tất cả các nhãn trong thư mục data/users
    users_dir = os.path.join('data', 'users')
    labels = [name for name in os.listdir(users_dir) if os.path.isdir(os.path.join(users_dir, name))]

    # Sử dụng camera RealSense để quét
    camera = RealSenseCameraNew()

    # Gọi hàm đánh giá
    evaluate_models(camera, knn_model, svm_model, input_label, labels)
