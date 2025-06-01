import tkinter as tk
from tkinter import ttk, messagebox
import pyautogui
import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self.root = tk.Tk()
        self.root.title("AI Trainer for Match-3")
        self.root.geometry("800x600")
        
        # Инициализация модели
        self.model = None
        self.target_model = None
        self.optimizer = None  
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.gamma = 0.95
        self.epsilon_decay = 0.999
        self.update_target_every = 1000
        self.replay_ratio = 2
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.steps = 0
        self.current_score = 0
        self.is_running = False
        self.penalty_counter = 0
        self.max_penalty_steps = 5
        self.penalty_value = -50
        self.best_score = 0
        self.generate_valid_actions()
        self.num_actions = len(self.valid_actions)
        
        self.create_widgets()
        self.init_visualization()
        self.build_model()
        

        self.root.bind('<KeyPress-q>', self.stop_training)
        

        self.scores = []
        self.total_reward = 0
        self.episode = 0

    def create_widgets(self):

        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.btn_board = ttk.Button(control_frame, text="1. Сохранить игровое поле", command=self.set_board)
        self.btn_board.pack(side=tk.LEFT, padx=5)
        
        self.btn_score = ttk.Button(control_frame, text="2. Сохранить область счета", command=self.set_score)
        self.btn_score.pack(side=tk.LEFT, padx=5)
        
        self.btn_train = ttk.Button(control_frame, text="3. Начать обучение", command=self.start_training)
        self.btn_train.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(control_frame, text="Стоп (Q)", command=self.stop_training)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        

        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.lbl_board = ttk.Label(status_frame, text="Игровое поле: не задано")
        self.lbl_board.pack(side=tk.LEFT, padx=10)
        
        self.lbl_score = ttk.Label(status_frame, text="Область счета: не задана")
        self.lbl_score.pack(side=tk.LEFT, padx=10)
        
        self.lbl_status = ttk.Label(status_frame, text="Статус: Ожидание настроек")
        self.lbl_status.pack(side=tk.LEFT, padx=10)

    def init_visualization(self):

        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot([], [])
        self.ax.set_xlabel('Шаги')
        self.ax.set_ylabel('Счет')
        self.ax.set_title('Прогресс обучения')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        
        stats_frame = ttk.Frame(self.root)
        stats_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.lbl_epsilon = ttk.Label(stats_frame, text="Epsilon: 1.0")
        self.lbl_epsilon.pack(side=tk.LEFT, padx=10)
        
        self.lbl_reward = ttk.Label(stats_frame, text="Total Reward: 0")
        self.lbl_reward.pack(side=tk.LEFT, padx=10)
        
        self.lbl_steps = ttk.Label(stats_frame, text="Steps: 0")
        self.lbl_steps.pack(side=tk.LEFT, padx=10)
    
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
        
  
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        try:
            text = pytesseract.image_to_string(img, config='--psm 7 digits')
            return int(''.join(filter(str.isdigit, text)))
        except:
            return 0
    def generate_valid_actions(self):
        self.valid_actions = []
    

        for row in range(10):
            for col in range(9):
                self.valid_actions.append(row * 9 + col)
    
 
        for row in range(9):
            for col in range(10):
                self.valid_actions.append(90 + row * 10 + col)
    
        print(f"Всего допустимых действий: {len(self.valid_actions)}")
    
    def build_model(self):
        inputs = layers.Input(shape=(10, 10, 1))
    
        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(self.num_actions, activation='linear')(x)  # Используем num_actions
    
        self.model = models.Model(inputs, outputs)
    
        self.optimizer = optimizers.Adam(learning_rate=float(0.001))
    
        self.model.compile(
            optimizer=self.optimizer,
            loss='huber'
        )
    
        self.target_model = models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        
        
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_current_state(self):
        x, y, w, h = self.board_region
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        state = np.zeros((10,10), dtype=int)
        cell_w = w // 10
        cell_h = h // 10
        
        game_colors_bgr = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0),
            (0, 255, 255), (255, 0, 255), (255, 255, 0)
        ]
        
        for r in range(10):
            for c in range(10):
                cell = img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                avg_color = np.mean(cell, axis=(0,1))
                distances = [np.linalg.norm(avg_color - color) for color in game_colors_bgr]
                state[r,c] = np.argmin(distances)
        
        return (state.reshape(10,10,1) - 2.5) / 2.5
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.valid_actions)
        else:
  
            state_input = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state_input, verbose=0)[0]
            valid_q = [q_values[i] for i in self.valid_actions]
            return self.valid_actions[np.argmax(valid_q)]
        
        
    def get_cell_center(self, row, col):
        x = self.board_region[0] + col * (self.board_region[2]/10) + (self.board_region[2]/20)
        y = self.board_region[1] + row * (self.board_region[3]/10) + (self.board_region[3]/20)
        return (x, y)

    def perform_action(self, action):
        try:

            if action < 90:
                row = action // 9
                col = action % 9
                pos1 = (row, col)
                pos2 = (row, col + 1)
            else:
                action -= 90
                row = action // 10
                col = action % 10
                pos1 = (row, col)
                pos2 = (row + 1, col)
        

            x1, y1 = self.get_cell_center(*pos1)
            x2, y2 = self.get_cell_center(*pos2)
        
            pyautogui.moveTo(x1, y1, duration=0.02)
            pyautogui.click()
            time.sleep(0.02)
            pyautogui.moveTo(x2, y2, duration=0.02)
            pyautogui.click()
            time.sleep(0.2)
        
            new_score = self.read_score()
            reward = new_score - self.current_score
            self.current_score = new_score
        
            return reward
        
        except Exception as e:
            print(f"Ошибка выполнения действия: {e}")
            return self.penalty_value * 2
        
    def start_training(self):
        if not hasattr(self, 'board_region') or not hasattr(self, 'score_region'):
            print("[Ошибка] Сначала задайте игровое поле и счет!")
            return

        self.is_running = True
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()
        print("[Система] Обучение начато")
    
    def calculate_reward(self, prev_state, new_state):
        prev_matches = self.find_matches(prev_state)
        new_matches = self.find_matches(new_state)
    
        reward = 0
        reward += len(new_matches) * 10
        reward += (len(new_matches) - len(prev_matches)) * 5
    
        if len(new_matches) >= 4:
            reward += 20
        if len(new_matches) >= 5:
            reward += 50

        if len(new_matches) == 0:
            reward -= 15
    
        potential = self.analyze_potential_moves(new_state)
        reward += potential * 3
    
        return reward

    def analyze_potential_moves(self, state):
        potential = 0
        for r in range(10):
            for c in range(10):
                if c < 9 and state[r][c] == state[r][c+1]:
                    potential +=1
                if r < 9 and state[r][c] == state[r+1][c]:
                    potential +=1
        return potential
    
    def training_loop(self):
        last_score = self.current_score
        no_progress_steps = 0
        
        while self.is_running:
            state = self.get_current_state()
            action = self.select_action(state)
            reward = self.perform_action(action)
            
            if reward <= 0:
                no_progress_steps += 1
                if no_progress_steps >= self.max_penalty_steps:
                    reward += self.penalty_value
                    no_progress_steps = 0
                    print("Наказание!")
            else:
                no_progress_steps = 0

            next_state = self.get_current_state()
            
            self.memory.append((state, action, reward, next_state, False))
            self.total_reward += reward
            
            if len(self.memory) >= self.batch_size:
                self.train_on_batch()
            
            self.update_visualization()
            self.root.update()
            
            time.sleep(0.01)
            
    def update_visualization(self):
        self.scores.append(self.current_score)
        self.line.set_data(range(len(self.scores)), self.scores)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        
        self.lbl_epsilon.config(text=f"Epsilon: {self.epsilon:.4f}")
        self.lbl_reward.config(text=f"Total Reward: {self.total_reward}")
        self.lbl_steps.config(text=f"Steps: {self.steps}")
            
    def save_model(self, path='match3_model.keras'):
        try:
            self.model.save(path)
            print(f"[Система] Модель сохранена в {path}")
        except Exception as e:
            print(f"[Ошибка] Не удалось сохранить модель: {e}")

    def load_model(self, path='match3_model.keras'):
        try:
            self.model = tf.keras.models.load_model(path)
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
            print("[Система] Предыдущая модель загружена")
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            self.build_model()
            
    def train_on_batch(self):
        if len(self.memory) < self.batch_size:
            return
    
        total_loss = 0
        for _ in range(self.replay_ratio):
            batch = random.sample(self.memory, self.batch_size)
        
            states = np.array([x[0] for x in batch])
            actions = np.array([x[1] for x in batch])
            rewards = np.array([x[2] for x in batch])
            next_states = np.array([x[3] for x in batch])
        
            target_q = self.model.predict(states, verbose=0)
            next_q = self.target_model.predict(next_states, verbose=0)

            for i in range(self.batch_size):
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

            history = self.model.fit(
                states, 
                target_q,
                batch_size=64,
                epochs=1,
                verbose=0
            )
            total_loss += history.history['loss'][0]
    
        if total_loss/self.replay_ratio < 10:
            current_lr = self.model.optimizer.learning_rate.numpy()
            new_lr = max(current_lr * 0.999, 1e-5)
            self.model.optimizer.learning_rate.assign(new_lr)
    
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_model()
    
        if self.current_score > self.best_score:
            self.best_score = self.current_score
            self.save_model('best_model.keras')
    
    def stop_training(self, event=None):
        if self.is_running:
            self.is_running = False
            self.save_model()
            print("\n[Система] Обучение остановлено")
                 
            
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    monitor = GameMonitor()
    monitor.run()