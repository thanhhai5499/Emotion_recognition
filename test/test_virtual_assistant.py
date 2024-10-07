import sys
import os

# Thêm thư mục 'src' vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from virtual_assistant import VirtualAssistant

if __name__ == "__main__":
    # Khởi tạo đối tượng trợ lý ảo
    assistant = VirtualAssistant()

    # Bắt đầu lắng nghe từ khóa
    print("Trợ lý ảo đang lắng nghe từ khóa 'hello'...")

    # Bắt đầu lắng nghe và in ra những gì hệ thống nhận được
    assistant.start_listening()
