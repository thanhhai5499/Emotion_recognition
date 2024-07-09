import serial
import threading

class ArduinoReader:
    def __init__(self, port, baud_rate=9600):
        self.ser = serial.Serial(port, baud_rate, timeout=1)
        self.running = True
        self.lock = threading.Lock()

    def read_data(self):
        with self.lock:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').rstrip()
                if line.isdigit():
                    return int(line)
        return None

    def stop(self):
        with self.lock:
            self.running = False
            self.ser.close()
