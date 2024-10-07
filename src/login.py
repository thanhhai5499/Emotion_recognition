import os
import time
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk, ImageOps
import cv2
from src.camera import RealSenseCameraNew
from src.processor import EmotionRecognitionProcessor
from src.face_recognition import FaceRecognition

class FaceRecognitionLoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Đăng Nhập Hệ Thống")
        
        # Thiết lập chế độ full-screen
        self.root.attributes('-fullscreen', True)

        # Load logo image for icon and top-left corner display
        self.icon_image = Image.open("assets/logo.png")
        self.icon_image_resized = self.icon_image.resize((128, 128), Image.LANCZOS)  # Tăng kích thước logo
        self.icon_photo = ImageTk.PhotoImage(self.icon_image_resized)
        self.root.iconphoto(False, self.icon_photo)

        self.logo_label = Label(root)
        self.logo_label.place(x=20, y=20)  # Điều chỉnh vị trí logo
        self.display_large_logo()

        self.title_label = Label(root, text="HỆ THỐNG GIÁM SÁT QUÁ TRÌNH TẬP LUYỆN", font=("Helvetica", 24))
        self.title_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        self.message_label = Label(root, text="Nhận diện khuôn mặt để đăng nhập", font=("Helvetica", 24))  # Tăng kích thước chữ
        self.message_label.place(relx=0.5, rely=0.16, anchor=tk.CENTER)

        # Khung video giữ nguyên tỷ lệ 16:9
        self.video_frame = Label(root, width=1280, height=720, bg='black')  # Tạo khung với màu nền đen để giữ vị trí video
        self.video_frame.place(relx=0.5, rely=0.55, anchor=tk.CENTER)  # Căn giữa màn hình

        # Thêm label cho thông báo thành công
        self.success_message_label = Label(root, text="", font=("Helvetica", 20), fg="green")
        self.success_message_label.place(relx=0.5, rely=0.92, anchor=tk.CENTER)  # Đặt phía dưới camera

        # Nút "Exit" ở góc dưới bên phải
        self.exit_button = Button(root, text="Exit", font=("Helvetica", 18), command=self.exit_full_screen)
        self.exit_button.place(relx=0.98, rely=0.98, anchor=tk.SE)  # Đặt ở góc dưới bên phải

        self.camera = RealSenseCameraNew()
        self.processor = EmotionRecognitionProcessor()
        self.face_recognition = FaceRecognition()
        self.start_time = time.time()
        self.recognized_user = None

        # Kiểm tra xem mô hình đã tồn tại chưa
        if not os.path.exists('models/knn_model.pkl') or not os.path.exists('models/svm_model.pkl'):
            self.show_warning_message("Không tìm thấy dữ liệu khuôn mặt. Cần thu thập dữ liệu.")
            self.root.after(3000, self.start_capture)
        else:
            self.start_camera_preview()

    def display_large_logo(self):
        large_logo_image = self.icon_image.resize((128, 128), Image.LANCZOS)  # Điều chỉnh kích thước logo
        large_logo_photo = ImageTk.PhotoImage(large_logo_image)
        self.logo_label.config(image=large_logo_photo)
        self.logo_label.image = large_logo_photo

    def start_camera_preview(self):
        """Mở camera nhưng đợi 8 giây trước khi quét nhận diện khuôn mặt"""
        self.start_time = time.time()
        self.message_label.config(text="NHẬN DIỆN KHUÔN MẶT ĐỂ ĐĂNG NHẬP.")
        
        # Cập nhật khung hình camera trong suốt 8 giây nhưng không quét
        self.update_camera_preview()
        self.root.after(8000, self.update_video)  # Sau 8 giây, bắt đầu quét khuôn mặt

    def update_camera_preview(self):
        """Hiển thị khung hình camera mà không quét khuôn mặt"""
        ret, color_image, depth_image = self.camera.get_frames()
        if ret:
            cv2image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)

            # Giữ nguyên chất lượng Full HD (1920x1080)
            img_fullhd = img.resize((1920, 1080))

            # Sử dụng letterboxing để giữ tỷ lệ 16:9 khi hiển thị trong khung 1280x720
            img_letterboxed = ImageOps.fit(img_fullhd, (1280, 720), Image.LANCZOS, centering=(0.5, 0.5))

            imgtk_resized = ImageTk.PhotoImage(image=img_letterboxed)
            self.video_frame.imgtk = imgtk_resized
            self.video_frame.configure(image=imgtk_resized)

        self.root.after(50, self.update_camera_preview)  # Tiếp tục hiển thị khung hình camera mỗi 50ms

    def update_video(self):
        """Sau 8 giây, bắt đầu quét nhận diện khuôn mặt"""
        if time.time() - self.start_time > 10 and self.recognized_user is None:
            self.show_warning_message("Không nhận diện được khuôn mặt. Cần thu thập dữ liệu.")
            self.root.after(1000, self.start_capture)
            return

        ret, color_image, depth_image = self.camera.get_frames()
        if ret:
            frame_with_landmarks, faces, _ = self.processor.process_frame(color_image)
            if len(faces) > 0:
                user = self.face_recognition.recognize_user(color_image)
                if user:
                    self.recognized_user = user
                    self.show_success_message()
                    return
            
            cv2image = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)

            # Giữ nguyên chất lượng Full HD (1920, 1080)
            img_fullhd = img.resize((1920, 1080))

            # Sử dụng letterboxing để giữ tỷ lệ 16:9 khi hiển thị trong khung 1280x720
            img_letterboxed = ImageOps.fit(img_fullhd, (1280, 720), Image.LANCZOS, centering=(0.5, 0.5))

            imgtk_resized = ImageTk.PhotoImage(image=img_letterboxed)
            self.video_frame.imgtk = imgtk_resized
            self.video_frame.configure(image=imgtk_resized)
        
        self.root.after(50, self.update_video)

    def show_warning_message(self, message):
        """Hiển thị thông báo cảnh báo ở giữa phía dưới"""
        self.success_message_label.config(text=message, fg="red")

    def show_success_message(self):
        """Hiển thị thông báo thành công và tự động chuyển trang sau 5 giây"""
        self.success_message_label.config(text=f"Đăng nhập thành công. Chào mừng {self.recognized_user}", fg="green")
        self.root.after(5000, self.navigate_to_emotion_recognition)  # Chuyển trang sau 10 giây

    def start_capture(self):
        self.root.destroy()
        from data_collection import UserDataCollectionApp
        data_collection_app = tk.Tk()
        UserDataCollectionApp(data_collection_app)
        data_collection_app.mainloop()

    def navigate_to_emotion_recognition(self):
        self.root.destroy()
        from gui import EmotionRecognitionApp
        main_app = tk.Tk()
        EmotionRecognitionApp(main_app)
        main_app.mainloop()

    def destroy(self):
        self.camera.release()

    def exit_full_screen(self, event=None):
        """Thoát chế độ toàn màn hình khi nhấn nút Exit hoặc phím ESC"""
        self.root.attributes('-fullscreen', False)
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionLoginApp(root)
    root.mainloop()
