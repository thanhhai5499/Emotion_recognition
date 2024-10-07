import os
import time
import tkinter as tk
from tkinter import Label, Entry, Button
from PIL import Image, ImageTk, ImageOps
import cv2
from threading import Thread
from src.camera import RealSenseCameraNew
from src.processor import EmotionRecognitionProcessor
from src.face_recognition import FaceRecognition

class UserDataCollectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thu Thập Dữ Liệu")
        
        # Thiết lập chế độ full-screen
        self.root.attributes('-fullscreen', True)

        # Logo
        self.icon_image = Image.open("assets/logo.png")
        self.icon_image_resized = self.icon_image.resize((90, 90), Image.LANCZOS)
        self.icon_photo = ImageTk.PhotoImage(self.icon_image_resized)
        self.root.iconphoto(False, self.icon_photo)

        self.logo_label = Label(root)
        self.logo_label.place(x=20, y=20)
        self.display_large_logo()

        # Tiêu đề chính
        self.title_label = Label(root, text="HỆ THỐNG GIÁM SÁT QUÁ TRÌNH TẬP LUYỆN", font=("Helvetica", 24))
        self.title_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Tiêu đề phụ
        self.subtitle_label = Label(root, text="Thu Thập Dữ Liệu", font=("Helvetica", 22))
        self.subtitle_label.place(relx=0.5, rely=0.16, anchor=tk.CENTER)

        # Khung video giữ nguyên tỷ lệ 16:9
        self.video_frame = Label(root, width=1280, height=720, bg='black')
        self.video_frame.place(relx=0.4, rely=0.55, anchor=tk.CENTER)

        # Nhập tên và tuổi
        self.name_label = Label(root, text="Tên:", font=("Helvetica", 18))
        self.name_label.place(relx=0.77, rely=0.25, anchor=tk.CENTER)
        self.name_entry = Entry(root, font=("Helvetica", 18))
        self.name_entry.place(relx=0.87, rely=0.25, anchor=tk.CENTER)

        self.age_label = Label(root, text="Tuổi:", font=("Helvetica", 18))
        self.age_label.place(relx=0.77, rely=0.3, anchor=tk.CENTER)
        self.age_entry = Entry(root, font=("Helvetica", 18))
        self.age_entry.place(relx=0.87, rely=0.3, anchor=tk.CENTER)

        # Nút bắt đầu quay
        self.capture_button = Button(root, text="Bắt đầu quay", font=("Helvetica", 18), command=self.start_capture)
        self.capture_button.place(relx=0.85, rely=0.35, anchor=tk.CENTER)

        # Nút thoát
        self.exit_button = Button(root, text="Thoát", font=("Helvetica", 18), command=self.exit_full_screen)
        self.exit_button.place(relx=0.98, rely=0.98, anchor=tk.SE)

        # Thêm một Label cho thông báo dưới nút "Bắt đầu quay"
        self.message_label = Label(root, text="", font=("Helvetica", 18))
        self.message_label.place(relx=0.85, rely=0.42, anchor=tk.CENTER)  # Đặt ở dưới nút bắt đầu quay

        # Camera và bộ xử lý cảm xúc
        self.camera = RealSenseCameraNew()
        self.processor = EmotionRecognitionProcessor()
        self.capturing = False

        self.update_video()

    def display_large_logo(self):
        large_logo_image = self.icon_image.resize((90, 90), Image.LANCZOS)
        large_logo_photo = ImageTk.PhotoImage(large_logo_image)
        self.logo_label.config(image=large_logo_photo)
        self.logo_label.image = large_logo_photo

    def update_video(self):
        if not self.capturing:
            ret, color_image, depth_image = self.camera.get_frames()
            if ret:
                # Giữ nguyên phần xử lý khung hình cũ
                frame_with_landmarks, _, _ = self.processor.process_frame(color_image)
                cv2image = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)

                # Giữ nguyên chất lượng Full HD (1920x1080) cho video gốc
                img_fullhd = img.resize((1920, 1080), Image.LANCZOS)

                # Hiển thị khung camera với kích thước 1280x720, sử dụng letterboxing để giữ tỷ lệ
                img_resized = img_fullhd.resize((1280, 720), Image.LANCZOS)
                imgtk_resized = ImageTk.PhotoImage(image=img_resized)

                # Cập nhật khung hiển thị video
                self.video_frame.imgtk = imgtk_resized
                self.video_frame.configure(image=imgtk_resized)
        
        self.root.after(50, self.update_video)

    def start_capture(self):
        name = self.name_entry.get().strip()
        age = self.age_entry.get().strip()
        
        if not name or not age:
            self.show_message("Tên và tuổi không được để trống!", "red")
            return

        self.capturing = True
        self.capture_button.config(state="disabled")
        
        user_folder = f'{name}_{age}'
        user_data_dir = os.path.join('data', 'users', user_folder)
        os.makedirs(user_data_dir, exist_ok=True)

        capture_thread = Thread(target=self.capture_video, args=(user_data_dir,))
        capture_thread.start()

    def capture_video(self, user_data_dir):
        start_time = time.time()
        frame_count = 0
        while time.time() - start_time < 5:
            ret, color_image, depth_image = self.camera.get_frames()
            if ret:
                gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                frame_path = os.path.join(user_data_dir, f'frame_{frame_count}.png')
                cv2.imwrite(frame_path, gray_frame)
                frame_count += 1
        self.camera.release()
        self.train_and_return()

    def train_and_return(self):
        training_thread = Thread(target=self.train_model)
        training_thread.start()

    def train_model(self):
        try:
            FaceRecognition.train_face_recognition_model('data/users', 'models')
            self.show_message("Thu Thập Thành Công!", "green")
            self.root.after(3000, self.navigate_to_login)  # Quay về trang đăng nhập sau 3 giây
        except Exception as e:
            self.show_message(f"Thu Thập Thất Bại: {str(e)}", "red")
            self.root.after(3000, self.reset_capture)  # Bắt đầu lại thu thập sau 3 giây

    def show_message(self, message, color):
        """Hiển thị thông báo với màu sắc"""
        self.message_label.config(text=message, fg=color)

    def reset_capture(self):
        """Reset lại giao diện để người dùng có thể thu thập lại dữ liệu"""
        self.capturing = False
        self.capture_button.config(state="normal")
        self.show_message("", "")  # Xóa thông báo

    def navigate_to_login(self):
        self.root.destroy()
        from login import FaceRecognitionLoginApp
        login_app = tk.Tk()
        FaceRecognitionLoginApp(login_app)
        login_app.mainloop()

    def exit_full_screen(self, event=None):
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = UserDataCollectionApp(root)
    root.mainloop()
