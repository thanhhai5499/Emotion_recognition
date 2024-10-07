import speech_recognition as sr
import time
import threading
from gtts import gTTS
import sys
import pygame

class VirtualAssistant:
    def __init__(self, heart_rate_value, emotion_value):
        self.heart_rate_value = heart_rate_value
        self.emotion_value = emotion_value
        self.last_abnormal_time = None
        self.last_heart_rate_alert_time = None
        self.keyword_detected = False
        self.running = False
        self.lock = threading.Lock()  # Sử dụng khóa để tránh truy cập đồng thời vào file
        self.speaking = False  # Trạng thái để kiểm soát việc phát âm thanh

    def speak(self, text):
        """Phát âm thanh từ text bằng gTTS và pygame"""
        def play_sound():
            with self.lock:  # Khóa để tránh truy cập đồng thời vào file
                try:
                    self.speaking = True  # Đặt trạng thái đang nói
                    output_path = "output.mp3"
                    tts = gTTS(text=text, lang="vi")
                    tts.save(output_path)
                    print(f"Đã lưu file âm thanh tại: {output_path}")

                    # Khởi tạo pygame và phát âm thanh
                    pygame.mixer.init()
                    pygame.mixer.music.load(output_path)
                    pygame.mixer.music.play()

                    # Đợi cho đến khi âm thanh phát xong
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)

                    # Dừng pygame mixer để giải phóng file
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()

                    print("File âm thanh đã phát xong.")

                    # Reset file về trạng thái rỗng sau khi phát xong
                    with open(output_path, 'wb') as f:
                        f.truncate(0)  # Làm rỗng file sau khi phát xong
                    print("File âm thanh đã được reset về rỗng sau khi phát xong.")
                except Exception as e:
                    print(f"Lỗi khi phát âm thanh: {e}")
                finally:
                    # Đợi 10 giây trước khi cho phép nói tiếp
                    time.sleep(10)
                    self.speaking = False  # Đặt lại trạng thái sau 10 giây

        # Khởi chạy phát âm thanh trên luồng riêng
        play_thread = threading.Thread(target=play_sound)
        play_thread.daemon = True
        play_thread.start()

    def listen_for_keyword(self):
        """Lắng nghe từ khóa trong một luồng riêng"""
        while True:
            with sr.Microphone() as source:
                print("Trợ lý ảo đang nghe...")
                self.recognizer = sr.Recognizer()
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)

            try:
                text = self.recognizer.recognize_google(audio, language="vi-VN")
                print(f"Người dùng nói: {text}")
                if "hello" in text.lower():
                    self.speak("Xin chào bạn, Hôm nay bạn thấy thế nào?, Chúc bạn một ngày tốt lành!, Bạn có cần tôi giúp gì không?")
                    self.keyword_detected = True
                    self.start_background_tasks()
                    break
                else:
                    print("Từ khóa chưa khớp. Tiếp tục lắng nghe...")
            except sr.UnknownValueError:
                print("Không hiểu được âm thanh.")
            except sr.RequestError as e:
                print(f"Lỗi yêu cầu từ Google Speech Recognition service: {e}")

            time.sleep(1)

    def start_listening(self):
        """Khởi chạy quá trình lắng nghe từ khóa trên luồng riêng"""
        threading.Thread(target=self.listen_for_keyword, daemon=True).start()

    def start_background_tasks(self):
        """Bắt đầu các tác vụ kiểm tra nhịp tim và điều kiện cảm xúc"""
        if not self.running:
            self.running = True
            threading.Thread(target=self.run_check_heart_rate, daemon=True).start()
            threading.Thread(target=self.run_check_conditions, daemon=True).start()

    def check_conditions(self):
        """Kiểm tra nếu cảm xúc bất thường và nhịp tim vượt ngưỡng 60-100 trong vòng 5s. Nếu liên tục bất thường trong vòng 10 giây thì đóng chương trình."""
        if self.keyword_detected and not self.speaking:  # Chỉ cho phép kiểm tra khi không đang nói
            try:
                emotion = self.emotion_value.cget("text")
                heart_rate = int(self.heart_rate_value.cget("text"))

                if (heart_rate < 40 or heart_rate > 140) and emotion == "Bất thường":
                    current_time = time.time()
                    # Nếu không có thời gian bắt đầu hoặc là bắt đầu một chu kỳ mới
                    if not self.last_abnormal_time:
                        self.last_abnormal_time = current_time
                        self.speak("Cảnh báo: Nhịp tim không ổn định và cảm xúc bất thường. Tôi sẽ liên hệ với bác sĩ ngay. Tôi sẽ dừng chương trình cho bạn")
                    # Nếu phát hiện liên tục trong 10 giây
                    elif current_time - self.last_abnormal_time >= 10:
                        self.speak("Dừng bài tập ngay, chương trình sẽ đóng lại.")
                        time.sleep(5)
                        sys.exit(0)
                else:
                    # Nếu nhịp tim và cảm xúc trở lại bình thường, reset thời gian và không nói cảnh báo nữa
                    if self.last_abnormal_time:
                        threading.Thread(target=lambda: time.sleep(10), daemon=True).start()
                    self.last_abnormal_time = None
            except ValueError:
                pass
        

    def check_heart_rate(self):
        """Kiểm tra nếu nhịp tim vượt ngưỡng 60-100 trong vòng 10s"""
        if self.keyword_detected and not self.speaking:  # Chỉ cho phép kiểm tra khi không đang nói
            try:
                heart_rate = int(self.heart_rate_value.cget("text"))
                # Lấy giá trị cảm xúc từ emotion_value config ở GUI
                emotion = self.emotion_value.cget("text")
                if heart_rate < 40 or heart_rate > 140:
                    current_time = time.time()
                    if not self.last_heart_rate_alert_time or (current_time - self.last_heart_rate_alert_time >= 10):
                        if emotion == "Bất thường":
                            self.check_conditions()
                        else:
                            self.speak("Nhịp tim không ổn định. Bạn có mệt không, chúng ta nghỉ chút nhé.")
                        self.last_heart_rate_alert_time = current_time
            except ValueError:
                pass

