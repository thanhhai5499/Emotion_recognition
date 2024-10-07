import os
import cv2
import numpy as np
import joblib
from sklearn import neighbors
from sklearn.svm import SVC
import dlib

# Sử dụng mô hình landmark khuôn mặt của dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")  # Đường dẫn tới mô hình dlib landmark

def load_image_file(path):
    """ Sử dụng OpenCV để tải ảnh """
    image = cv2.imread(path)  # Đọc ảnh bằng OpenCV
    if image is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại {path}")
    return image

class FaceRecognition:
    @staticmethod
    def train_face_recognition_model(data_dir, model_dir):
        X = []
        y = []

        trained_users = set()

        # Kiểm tra danh sách những người đã được huấn luyện trước đó
        trained_users_path = os.path.join(model_dir, 'trained_users.txt')
        if os.path.exists(trained_users_path):
            with open(trained_users_path, 'r') as f:
                trained_users = set(f.read().splitlines())

        # Duyệt qua tất cả các thư mục con trong data_dir
        for root, dirs, files in os.walk(data_dir):
            label = os.path.basename(root)
            if label in trained_users:
                print(f"Người dùng {label} đã được huấn luyện. Bỏ qua.")
                continue  # Bỏ qua người dùng đã huấn luyện

            for file in files:
                if file.endswith(".png"):
                    path = os.path.join(root, file)
                    
                    # Tiền xử lý ảnh và kiểm tra chất lượng khuôn mặt
                    landmarks = FaceRecognition.extract_landmarks(path)
                    if landmarks is not None:
                        X.append(landmarks)
                        y.append(label)

        # Chuyển đổi danh sách thành numpy arrays
        if X and y:
            X = np.array(X)
            y = np.array(y)

            # Huấn luyện mô hình k-NN với k=3 dựa trên các đặc điểm khuôn mặt
            knn = neighbors.KNeighborsClassifier(n_neighbors=3)
            knn.fit(X, y)

            # Huấn luyện mô hình SVM
            svm = SVC(kernel='linear', probability=True)
            svm.fit(X, y)

            # Lưu cả hai mô hình
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            knn_model_path = os.path.join(model_dir, "knn_model.pkl")
            svm_model_path = os.path.join(model_dir, "svm_model.pkl")
            joblib.dump(knn, knn_model_path)
            joblib.dump(svm, svm_model_path)
            print("Huấn luyện hoàn tất và mô hình đã được lưu tại:", knn_model_path, "và", svm_model_path)

            # Cập nhật danh sách người dùng đã được huấn luyện
            with open(trained_users_path, 'a') as f:
                for label in np.unique(y):
                    f.write(f"{label}\n")

        else:
            print("Không tìm thấy dữ liệu mới để huấn luyện.")

    @staticmethod
    def extract_landmarks(image_path):
        """ Trích xuất 68 điểm đặc trưng khuôn mặt từ ảnh và chuẩn hóa """
        image = load_image_file(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if len(faces) == 0:
            return None
        
        # Trích xuất các landmark cho khuôn mặt đầu tiên được phát hiện
        face = faces[0]
        landmarks = predictor(gray, face)

        # Chuyển đổi các điểm đặc trưng thành một danh sách tọa độ
        landmark_points = []
        for i in range(0, 68):
            landmark_points.append((landmarks.part(i).x, landmarks.part(i).y))
        
        # Chuẩn hóa các tọa độ để loại bỏ tác động của vị trí và kích thước khuôn mặt
        normalized_landmarks = FaceRecognition.normalize_landmarks(landmark_points)
        
        # Làm phẳng danh sách các tọa độ để có thể đưa vào mô hình học máy
        flattened_landmarks = np.array(normalized_landmarks).flatten()
        return flattened_landmarks

    @staticmethod
    def normalize_landmarks(landmarks):
        """ Chuẩn hóa các tọa độ landmark bằng cách đưa về cùng tỷ lệ """
        # Chuyển đổi danh sách các tọa độ thành một numpy array
        landmarks = np.array(landmarks)

        # Tính trung tâm của khuôn mặt (giữa mắt)
        mean = np.mean(landmarks, axis=0)

        # Chuẩn hóa tọa độ bằng cách trừ đi trung tâm và chia cho khoảng cách chuẩn hóa
        normalized_landmarks = landmarks - mean
        max_distance = np.linalg.norm(normalized_landmarks, axis=1).max()
        normalized_landmarks /= max_distance

        return normalized_landmarks

    def __init__(self):
        self.model_path_knn = os.path.join('models', 'knn_model.pkl')
        self.model_path_svm = os.path.join('models', 'svm_model.pkl')
        self.knn_classifier = None
        self.svm_classifier = None
        if os.path.exists(self.model_path_knn) and os.path.exists(self.model_path_svm):
            self.knn_classifier = joblib.load(self.model_path_knn)
            self.svm_classifier = joblib.load(self.model_path_svm)
        else:
            print("Mô hình nhận diện khuôn mặt không tồn tại!")

    def recognize_user(self, image):
        if self.knn_classifier is None or self.svm_classifier is None:
            return None
        
        # Trích xuất các điểm đặc trưng từ khuôn mặt trong ảnh
        landmarks = self.extract_landmarks_from_frame(image)
        if landmarks is None:
            return None

        # Dự đoán bằng KNN và SVM
        knn_prediction = self.knn_classifier.predict([landmarks])
        svm_prediction = self.svm_classifier.predict([landmarks])
        
        # Sử dụng dự đoán từ KNN và SVM, có thể chọn cách kết hợp nếu cần
        return knn_prediction[0] if knn_prediction == svm_prediction else None

    def extract_landmarks_from_frame(self, frame):
        """ Trích xuất 68 điểm từ một khung hình và chuẩn hóa """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if len(faces) == 0:
            return None

        face = faces[0]
        landmarks = predictor(gray, face)
        landmark_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        
        # Chuẩn hóa các điểm đặc trưng
        normalized_landmarks = self.normalize_landmarks(landmark_points)
        return np.array(normalized_landmarks).flatten()
