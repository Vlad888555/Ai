import time
import random
import os
from collections import deque

import numpy as np
import cv2
from mss import mss
import pyautogui
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Константы окружения
REGION = {'top': 100, 'left': 200, 'width': 400, 'height': 400}
GRID_SIZE = (8, 8)
ACTION_SPACE = [(i, j, di, dj) for i in range(GRID_SIZE[0])
                                 for j in range(GRID_SIZE[1])
                                 for di, dj in [(1, 0), (0, 1)]]  # возможные обмены
MODEL_PATH = 'dqn_match3_model.h5'

# Гиперпараметры обучения
EPISODES = 1000
MAX_MEMORY = 2000
BATCH_SIZE = 32
GAMMA = 0.95
LEARNING_RATE = 0.001

class DQNAgent:
    def __init__(self, state_shape, action_size, model_path=MODEL_PATH):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = self._build_model()
        self.model_path = model_path
        # Попытка загрузить модель, если существует
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model.load_weights(self.model_path)
        else:
            print("No existing model found, training from scratch.")

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([b[0] for b in batch])
        next_states = np.array([b[3] for b in batch])
        q_next = self.model.predict(next_states, verbose=0)
        x, y = [], []
        for i, (state, action, reward, _, done) in enumerate(batch):
            target = reward
            if not done:
                target += GAMMA * np.amax(q_next[i])
            q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
            q_values[action] = target
            x.append(state)
            y.append(q_values)
        self.model.fit(np.array(x), np.array(y), epochs=1, verbose=0)

    def save_model(self):
        self.model.save_weights(self.model_path)
        print(f"Model saved to {self.model_path}")


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


def train():
    state_shape = GRID_SIZE
    agent = DQNAgent(state_shape, len(ACTION_SPACE))
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    for e in range(EPISODES):
        state = capture_state(REGION)
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state, epsilon)
            perform_action(action, REGION)
            next_state = capture_state(REGION)
            reward = get_reward(state, next_state)
            total_reward += reward
            done = total_reward < -50
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

        # Сохранение модели после каждой эпохи
        agent.save_model()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        print(f"Episode {e+1}/{EPISODES}, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    print("Starting training...")
    train()
