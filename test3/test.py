import tkinter as tk
from tkinter import messagebox
import threading

нужно настроить данные Reagion
# Подключаем все необходимые модули из твоего кода
import time
import random
import os
from collections import deque

import pytesseract
from PIL import Image

import numpy as np
import cv2
from mss import mss
import pyautogui
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Глобальные переменные
GRID_SIZE = (8, 8)
ACTION_SPACE = [(i, j, di, dj) for i in range(GRID_SIZE[0])
                                 for j in range(GRID_SIZE[1])
                                 for di, dj in [(1, 0), (0, 1)]]
MODEL_PATH = 'dqn_match3_model.h5'

# ----- Выбор области -----
def RegionSelector(prompt="Выделите область и отпустите кнопку мыши"):
    """
    Блокирует выполнение до тех пор, пока вы не выделите область мышью.
    Возвращает dict: {'top':…, 'left':…, 'width':…, 'height':…}
    """
    region = {}
    start = {'x': 0, 'y': 0}
    rect_id = {'id': None}

    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.attributes('-alpha', 0.3)
    root.configure(background='black')
    root.title(prompt)

    canvas = tk.Canvas(root, cursor='cross')
    canvas.pack(fill=tk.BOTH, expand=True)

    def on_mouse_down(event):
        start['x'] = canvas.canvasx(event.x)
        start['y'] = canvas.canvasy(event.y)
        rect_id['id'] = canvas.create_rectangle(
            start['x'], start['y'],
            start['x'], start['y'],
            outline='red', width=2
        )

    def on_mouse_drag(event):
        cur_x = canvas.canvasx(event.x)
        cur_y = canvas.canvasy(event.y)
        canvas.coords(rect_id['id'],
                      start['x'], start['y'],
                      cur_x,    cur_y)

    def on_mouse_up(event):
        end_x = canvas.canvasx(event.x)
        end_y = canvas.canvasy(event.y)
        left   = min(start['x'], end_x)
        top    = min(start['y'], end_y)
        width  = abs(start['x'] - end_x)
        height = abs(start['y'] - end_y)
        region.update({
            'top':    int(top),
            'left':   int(left),
            'width':  int(width),
            'height': int(height),
        })
        print("Selected region:", region)
        root.destroy()

    # привязка событий
    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>",    on_mouse_drag)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)
    root.bind('<Escape>', lambda e: root.destroy())

    return region

# ---- DQN Агент и обучение ----
class DQNAgent:
    def __init__(self, state_shape, action_size, model_path=MODEL_PATH):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = self._build_model()
        self.model_path = model_path
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model.load_weights(self.model_path)

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < 32:
            return
        batch = random.sample(self.memory, 32)
        states = np.array([b[0] for b in batch])
        next_states = np.array([b[3] for b in batch])
        q_next = self.model.predict(next_states, verbose=0)
        x, y = [], []
        for i, (state, action, reward, _, done) in enumerate(batch):
            target = reward
            if not done:
                target += 0.95 * np.amax(q_next[i])
            q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
            q_values[action] = target
            x.append(state)
            y.append(q_values)
        self.model.fit(np.array(x), np.array(y), epochs=1, verbose=0)

    def save_model(self):
        self.model.save_weights(self.model_path)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, GRID_SIZE)
    normalized = resized / 255.0
    return normalized

def capture_state(monitor):
    with mss() as sct:
        sct_img = np.array(sct.grab(monitor))
    return preprocess(sct_img)

def perform_action(action_idx, region):
    i, j, di, dj = ACTION_SPACE[action_idx]
    cell_w = region['width'] / GRID_SIZE[1]
    cell_h = region['height'] / GRID_SIZE[0]
    x1 = region['left'] + j * cell_w + cell_w / 2
    y1 = region['top'] + i * cell_h + cell_h / 2
    x2 = x1 + dj * cell_w
    y2 = y1 + di * cell_h
    pyautogui.moveTo(x1, y1, duration=0.1)
    pyautogui.dragTo(x2, y2, duration=0.2)
    time.sleep(0.2)

def get_reward(prev_state, current_state):
    return np.sum(current_state) - np.sum(prev_state)

def Region():
    global REGION
    

# Функция-обёртка для выбора области счёта


def train_model(episodes=1000, required_score=200, max_failures=3):
    agent = DQNAgent(GRID_SIZE, len(ACTION_SPACE))
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    failure_streak = 0

    for ep in range(1, episodes + 1):
        # стартовый счёт
        start_score = read_score() or 0

        state = capture_state(REGION)
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state, epsilon)
            perform_action(action, REGION)
            next_state = capture_state(REGION)

            current_score = read_score() or start_score
            delta = current_score - start_score
            reward = delta if delta != 0 else -1

            total_reward += reward
            done = total_reward < -50

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

        agent.save_model()

        gained = (read_score() or start_score) - start_score
        if gained < required_score:
            failure_streak += 1
            print(f"Episode {ep}: FAIL (gained {gained} < {required_score}), streak {failure_streak}")
        else:
            failure_streak = 0
            print(f"Episode {ep}: OK (gained {gained})")

        if failure_streak >= max_failures:
            print(f"Punishment triggered after {failure_streak} failures!")
            for _ in range(5):
                agent.remember(state, action, -10, state, True)
            failure_streak = 0

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Epsilon: {epsilon:.3f}\n")
        
def read_score():
    if SCORE_REGION is None:
        print("Сначала выберите область с отображением счета (кнопка «4. Выбрать область счета»)")
        return None
    # Делаем скриншот нужного фрагмента экрана
    img = pyautogui.screenshot(region=(
        SCORE_REGION['left'],
        SCORE_REGION['top'],
        SCORE_REGION['width'],
        SCORE_REGION['height'],
    ))
    # Конвертируем в оттенки серого и порог
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Запускаем OCR (psm 7 — одиночная строка)
    text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
    # Оставляем только цифры
    digits = ''.join(filter(str.isdigit, text))
    score = int(digits) if digits.isdigit() else None
    print(f"Распознанный счет: {score}")
    return score

def select_score_area():
    global SCORE_REGION
    selector = RegionSelector()      # тот же класс, что вы юзаете для выбора REGION
    SCORE_REGION = selector
    print("SCORE_REGION =", SCORE_REGION)


# ---- GUI ----
def run_gui():
    root = tk.Tk()
    root.title("Match-3 DQN Меню")
    root.geometry("300x200")

    def select_area():
        threading.Thread(target=RegionSelector).start()

    def train_thread():
        threading.Thread(target=train_model).start()

    def use_thread():
        threading.Thread(target=use_model).start()

    tk.Button(root, text="1. Выбрать область", command=select_area, height=2).pack(fill='x', pady=5)
    tk.Button(root, text="2. Обучить модель", command=train_thread, height=2).pack(fill='x', pady=5)
    tk.Button(root, text="3. Использовать без обучения", command=use_thread, height=2).pack(fill='x', pady=5)
    tk.Button(root, text="4. Выбрать область счета", command=select_score_area, height=2).pack(fill='x', pady=5)
    
    print(read_score)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
