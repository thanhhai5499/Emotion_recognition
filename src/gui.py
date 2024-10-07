import sys
import os
import threading  # Thêm threading để hỗ trợ chạy song song
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import time
from src.camera import RealSenseCameraNew
from src.processor import EmotionRecognitionProcessor
from src.arduino_reader import ArduinoReader
from src.virtual_assistant import VirtualAssistant  # Import Virtual Assistant

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận Diện Tình Trạng")
        
        self.root.attributes('-fullscreen', True)

        # Load logo image for icon and top-left corner display
        self.icon_image = Image.open("assets/logo.png")
        self.icon_image_resized = self.icon_image.resize((64, 64), Image.LANCZOS)
        self.icon_photo = ImageTk.PhotoImage(self.icon_image_resized)
        self.root.iconphoto(False, self.icon_photo)

        self.logo_label = Label(root)
        self.logo_label.pack(side="top", anchor="nw", padx=10, pady=10)
        self.display_large_logo()

        self.title_label = Label(root, text="HỆ THỐNG GIÁM SÁT QUÁ TRÌNH TẬP LUYỆN", font=("Helvetica", 24))
        self.title_label.pack(pady=20)

        self.video_frame = Label(root)
        self.video_frame.place(relx=0.05, rely=0.2, relwidth=0.4, relheight=0.5)

        self.depth_frame = Label(root)
        self.depth_frame.place(relx=0.55, rely=0.2, relwidth=0.4, relheight=0.5)

        self.special_message = Label(root, text="Thông báo", font=("Helvetica", 18), fg="orange", bg="white", anchor="w")
        self.special_message.place(relx=0.95, rely=0.15, anchor=tk.NE)

        self.emotion_label = Label(root, text="Cảm Xúc:", font=("Helvetica", 18), bg="light blue", anchor="w")
        self.emotion_label.place(relx=0.05, rely=0.75, relwidth=0.2)

        self.emotion_value = Label(root, text="", font=("Helvetica", 18), anchor="w")
        self.emotion_value.place(relx=0.25, rely=0.75, relwidth=0.2)

        self.rest_message = Label(root, text="Cảnh Báo:", font=("Helvetica", 18), bg="light blue", anchor="w")
        self.rest_message.place(relx=0.05, rely=0.8, relwidth=0.2)

        self.rest_value = Label(root, text="", font=("Helvetica", 18), fg="red", anchor="w")
        self.rest_value.place(relx=0.25, rely=0.8, relwidth=0.2)

        self.heart_rate_label = Label(root, text="Nhịp Tim:", font=("Helvetica", 18), bg="light blue", anchor="w")
        self.heart_rate_label.place(relx=0.55, rely=0.75, relwidth=0.2)

        self.heart_rate_value = Label(root, text="Đang đọc...", font=("Helvetica", 18), anchor="w")
        self.heart_rate_value.place(relx=0.75, rely=0.75, relwidth=0.2)

        self.reset_button = Button(root, text="Thoát", font=("Helvetica", 18), command=self.close_app)
        self.reset_button.place(relx=0.55, rely=0.8, relwidth=0.2)

        # Khởi tạo Virtual Assistant và chạy ngầm
        self.assistant = VirtualAssistant(self.heart_rate_value, self.emotion_value)


        # Chạy trợ lý ảo trên một luồng riêng
        assistant_thread = threading.Thread(target=self.assistant.start_listening, daemon=True)
        assistant_thread.start()
        
        try:
            self.camera = RealSenseCameraNew()
            self.processor = EmotionRecognitionProcessor()
            self.special_message.config(text="")
        except Exception as e:
            self.special_message.config(text="Không tìm thấy camera!")
            self.camera = None
            self.processor = None

        self.arduino_reader = None
        try:
            self.arduino_reader = ArduinoReader(port='COM9')
            self.special_message.config(text="")
        except Exception as e:
            if self.camera is None:
                self.special_message.config(text="Không tìm thấy camera và Arduino!")
            else:
                self.special_message.config(text="Không tìm thấy Arduino!")
            self.arduino_reader = None

        if self.camera and self.processor:
            self.update_video()

        if self.arduino_reader:
            self.update_heart_rate()

    def display_large_logo(self):
        large_logo_image = self.icon_image.resize((64, 64), Image.LANCZOS)
        large_logo_photo = ImageTk.PhotoImage(large_logo_image)
        self.logo_label.config(image=large_logo_photo)
        self.logo_label.image = large_logo_photo

    def close_app(self):
        self.root.quit()

    def update_video(self):
        if self.camera is None or self.processor is None:
            return

        ret, color_image, depth_image = self.camera.get_frames()
        if ret:
            current_time = time.time()
            if current_time - self.processor.last_update_time >= 0.05:
                frame_with_landmarks, faces, landmarks = self.processor.process_frame1(color_image)
                if len(faces) == 0:
                    self.special_message.config(text="Không tìm thấy khuôn mặt!")
                else:
                    self.special_message.config(text="")
                    emotion = self.processor.predict_emotion(color_image)
                    if emotion is not None:
                        emotion_text = self.processor.get_emotion_text(emotion)

                        # Kiểm tra trạng thái cảm xúc
                        if emotion_text == "Bất thường":
                            if self.processor.current_emotion == emotion_text:
                                # Nếu cảm xúc bất thường đã duy trì hơn 9 giây
                                if time.time() - self.processor.emotion_start_time >= 3:
                                    self.emotion_value.config(text="Bất thường")
                                    self.rest_value.config(text="Cần nghỉ ngơi")
                                # Nếu chưa tới 9 giây, vẫn hiện "Bình thường"
                                elif time.time() - self.processor.emotion_start_time < 2:
                                    self.emotion_value.config(text="Bình thường")
                            else:
                                # Khởi động thời gian khi phát hiện "Bất thường"
                                self.processor.current_emotion = emotion_text
                                self.processor.emotion_start_time = time.time()
                        else:
                            # Nếu cảm xúc là bình thường, reset lại thời gian và hiển thị "Bình thường"
                            self.processor.current_emotion = None
                            self.processor.emotion_start_time = None
                            self.emotion_value.config(text="Bình thường")
                            self.rest_value.config(text="")

                        self.processor.last_update_time = current_time

                # Hiển thị hình ảnh màu
                cv2image_color = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGBA)
                img_color = Image.fromarray(cv2image_color)
                img_color = img_color.resize((int(0.4 * self.root.winfo_width()), int(0.5 * self.root.winfo_height())))
                imgtk_color = ImageTk.PhotoImage(image=img_color)
                self.video_frame.imgtk = imgtk_color
                self.video_frame.configure(image=imgtk_color)

                # Hiển thị hình ảnh chiều sâu
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2image_depth = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGBA)
                img_depth = Image.fromarray(cv2image_depth)
                img_depth = img_depth.resize((int(0.4 * self.root.winfo_width()), int(0.5 * self.root.winfo_height())))
                imgtk_depth = ImageTk.PhotoImage(image=img_depth)
                self.depth_frame.imgtk = imgtk_depth
                self.depth_frame.configure(image=imgtk_depth)

        # Kiểm tra nhịp tim và cảm xúc trong khi update video
        self.assistant.check_heart_rate()
        self.assistant.check_conditions()

        self.root.after(50, self.update_video)


    def update_heart_rate(self):
        if self.arduino_reader:
            heart_rate = self.arduino_reader.read_data()
            if heart_rate is not None:
                self.heart_rate_value.config(text=str(heart_rate))
            self.root.after(1000, self.update_heart_rate)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
