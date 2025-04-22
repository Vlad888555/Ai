import tkinter as tk
from tkinter import ttk, messagebox
import pyautogui
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class RegionSelector(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.attributes('-fullscreen', True)
        self.attributes('-alpha', 0.3)
        self.configure(bg='black')
        self.attributes('-topmost', True)
        
        self.canvas = tk.Canvas(self, cursor='cross', bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.coords = None
        
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

    def on_press(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        self.rect = self.canvas.create_rectangle(0, 0, 0, 0, outline='#00ff00', width=2)

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x_root, event.y_root)

    def on_release(self, event):
        x1 = min(self.start_x, event.x_root)
        y1 = min(self.start_y, event.y_root)
        x2 = max(self.start_x, event.x_root)
        y2 = max(self.start_y, event.y_root)
        self.coords = (x1, y1, x2 - x1, y2 - y1)
        self.destroy()

class GameMonitor:
    def __init__(self):
        self.board_region = None
        self.score_region = None
        self.is_running = False
        
        self.root = tk.Tk()
        self.root.title("AI Trainer for Match-3")
        self.root.geometry("400x300")
        
        self.create_widgets()
        
    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.btn_board = ttk.Button(frame, text="1. Сохранить игровое поле", command=self.set_board)
        self.btn_board.pack(pady=5, fill=tk.X)
        
        self.btn_score = ttk.Button(frame, text="2. Сохранить область счета", command=self.set_score)
        self.btn_score.pack(pady=5, fill=tk.X)
        
        self.btn_train = ttk.Button(frame, text="3. Начать обучение", command=self.start_training)
        self.btn_train.pack(pady=20, fill=tk.X)
        
        self.lbl_status = ttk.Label(frame, text="Статус: Ожидание настроек")
        self.lbl_status.pack(pady=10)
        
        self.lbl_board = ttk.Label(frame, text="Игровое поле: не задано")
        self.lbl_board.pack()
        
        self.lbl_score = ttk.Label(frame, text="Область счета: не задана")
        self.lbl_score.pack()

    def set_board(self):
        selector = RegionSelector(self.root)
        self.root.wait_window(selector)
        if selector.coords:
            self.board_region = selector.coords
            self.lbl_board.config(text=f"Игровое поле: {self.board_region}")

    def set_score(self):
        selector = RegionSelector(self.root)
        self.root.wait_window(selector)
        if selector.coords:
            self.score_region = selector.coords
            self.lbl_score.config(text=f"Область счета: {self.score_region}")

    def read_score(self):
        if not self.score_region:
            return 0
            
        x, y, w, h = self.score_region
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        
        # Улучшаем изображение для OCR
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        try:
            text = pytesseract.image_to_string(img, config='--psm 7 digits')
            return int(''.join(filter(str.isdigit, text)))
        except:
            return 0

    def start_training(self):
        if not self.board_region or not self.score_region:
            messagebox.showerror("Ошибка", "Сначала задайте обе области!")
            return
            
        self.is_running = True
        self.lbl_status.config(text="Статус: Обучение запущено")
        
        # Здесь будет основной цикл обучения
        while self.is_running:
            current_score = self.read_score()
            print(f"Текущий счет: {current_score}")
            
            # TODO: Добавить логику ИИ
            # ...
            
            self.root.update()
            self.root.after(1000)  # Обновление каждую секунду

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    monitor = GameMonitor()
    monitor.run()