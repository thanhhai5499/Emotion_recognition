import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# main.py
import tkinter as tk
from login import FaceRecognitionLoginApp

def main():
    root = tk.Tk()
    app = FaceRecognitionLoginApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
