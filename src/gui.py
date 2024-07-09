import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import time
from src.camera import RealSenseCamera
from src.processor import EmotionRecognitionProcessor
from src.arduino_reader import ArduinoReader

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition")
        self.root.geometry("1000x600")  # Set fixed size for the window

        self.video_frame = Label(root)
        self.video_frame.place(x=10, y=10, width=480, height=360)

        self.depth_frame = Label(root)
        self.depth_frame.place(x=510, y=10, width=480, height=360)

        self.special_message = Label(root, text="Thông báo", font=("Helvetica", 16), fg="orange", bg="white", width=25, anchor="w")
        self.special_message.place(x=10, y=380)

        self.emotion_label = Label(root, text="Cảm Xúc:", font=("Helvetica", 16), bg="light blue", width=15, anchor="w")
        self.emotion_label.place(x=10, y=420)

        self.emotion_value = Label(root, text="", font=("Helvetica", 16), width=15, anchor="w")
        self.emotion_value.place(x=200, y=420)

        self.rest_message = Label(root, text="Cảnh Báo:", font=("Helvetica", 16), bg="light blue", width=15, anchor="w")
        self.rest_message.place(x=10, y=460)

        self.rest_value = Label(root, text="", font=("Helvetica", 16), fg="red", width=15, anchor="w")
        self.rest_value.place(x=200, y=460)

        self.heart_rate_label = Label(root, text="Nhịp Tim:", font=("Helvetica", 16), bg="light blue", width=15, anchor="w")
        self.heart_rate_label.place(x=10, y=500)

        self.heart_rate_value = Label(root, text="Đang đọc...", font=("Helvetica", 16), width=15, anchor="w")
        self.heart_rate_value.place(x=200, y=500)

        self.reset_button = Button(root, text="Reset", font=("Helvetica", 16), command=self.reset_app)
        self.reset_button.place(x=10, y=540)

        try:
            self.camera = RealSenseCamera()
            self.processor = EmotionRecognitionProcessor()
            self.special_message.config(text="")
        except Exception as e:
            self.special_message.config(text="Không tìm thấy camera!")
            self.camera = None
            self.processor = None

        self.arduino_reader = None
        try:
            self.arduino_reader = ArduinoReader(port='COM11')  # Thay thế 'COM3' bằng cổng serial của Arduino của bạn
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

    def reset_app(self):
        self.root.destroy()
        main()

    def update_video(self):
        if self.camera is None or self.processor is None:
            return

        color_frame, depth_frame = self.camera.get_frames()
        if color_frame is not None and depth_frame is not None:
            current_time = time.time()
            if current_time - self.processor.last_update_time >= 0.05:  # Quét khuôn mặt mỗi 50ms
                frame_with_landmarks, faces, landmarks = self.processor.process_frame(color_frame)
                if len(faces) == 0:
                    self.special_message.config(text="Không tìm thấy khuôn mặt!")
                else:
                    self.special_message.config(text="")
                    emotion = self.processor.predict_emotion(color_frame)
                    if emotion is not None:
                        emotion_text = self.processor.get_emotion_text(emotion)
                        self.emotion_value.config(text=emotion_text)

                        if emotion_text in ["Tức giận", "Ghê tởm", "Sợ hãi"]:
                            if self.processor.current_emotion == emotion_text:
                                if time.time() - self.processor.emotion_start_time >= 5:  # Cảnh báo khi cảm xúc quá 5s
                                    self.rest_value.config(text="Cần nghỉ ngơi")
                            else:
                                self.processor.current_emotion = emotion_text
                                self.processor.emotion_start_time = time.time()
                        else:
                            self.processor.current_emotion = None
                            self.processor.emotion_start_time = None
                            self.rest_value.config(text="")

                        self.processor.last_update_time = current_time

                cv2image_color = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGBA)
                img_color = Image.fromarray(cv2image_color)
                img_color = img_color.resize((480, 360))  # Giảm kích thước video
                imgtk_color = ImageTk.PhotoImage(image=img_color)
                self.video_frame.imgtk = imgtk_color
                self.video_frame.configure(image=imgtk_color)

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
                cv2image_depth = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGBA)
                img_depth = Image.fromarray(cv2image_depth)
                img_depth = img_depth.resize((480, 360))  # Giảm kích thước video RGB
                imgtk_depth = ImageTk.PhotoImage(image=img_depth)
                self.depth_frame.imgtk = imgtk_depth
                self.depth_frame.configure(image=imgtk_depth)

        self.root.after(50, self.update_video)  # Cập nhật video mỗi 50ms

    def update_heart_rate(self):
        if self.arduino_reader:
            heart_rate = self.arduino_reader.read_data()
            if heart_rate is not None:
                self.heart_rate_value.config(text=str(heart_rate))
            self.root.after(1000, self.update_heart_rate)  # Cập nhật nhịp tim mỗi giây

def main():
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
