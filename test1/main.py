import numpy as np
import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import matplotlib.pyplot as plt

###########################
# 1. Окружение "3 в ряд"  #
###########################

class Match3Env:
    def __init__(self, grid_size=6, n_types=4, max_steps=50, cell_size=50):
        self.grid_size = grid_size
        self.n_types = n_types
        self.max_steps = max_steps
        self.cell_size = cell_size
        self.actions = self._generate_actions()
        self.reset()
        
        pygame.init()
        self.screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
        pygame.display.set_caption("Match-3 Environment")

    def _generate_actions(self):

        actions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if j < self.grid_size - 1:  # swap вправо
                    actions.append(((i, j), (i, j+1)))
                if i < self.grid_size - 1:  # swap вниз
                    actions.append(((i, j), (i+1, j)))
        return actions

    def reset(self):
        self.grid = np.random.randint(0, self.n_types, (self.grid_size, self.grid_size))
        self.steps = 0
        return self._get_state()

    def _get_state(self):

        return np.copy(self.grid)

    def render(self):
        colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255)
        ]
        self.screen.fill((0, 0, 0))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                candy_type = self.grid[i, j]
                color = colors[candy_type % len(colors)]
                pygame.draw.rect(
                    self.screen,
                    color,
                    (j * self.cell_size, i * self.cell_size, self.cell_size - 2, self.cell_size - 2)
                )
        pygame.display.flip()

    def _check_matches(self):

        remove = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        for i in range(self.grid_size):
            count = 1
            for j in range(1, self.grid_size):
                if self.grid[i, j] == self.grid[i, j-1]:
                    count += 1
                else:
                    if count >= 3:
                        remove[i, j-count:j] = True
                    count = 1
            if count >= 3:
                remove[i, self.grid_size-count:self.grid_size] = True


        for j in range(self.grid_size):
            count = 1
            for i in range(1, self.grid_size):
                if self.grid[i, j] == self.grid[i-1, j]:
                    count += 1
                else:
                    if count >= 3:
                        remove[i-count:i, j] = True
                    count = 1
            if count >= 3:
                remove[self.grid_size-count:self.grid_size, j] = True

        return remove

    def _update_grid(self, remove):

        n_removed = np.sum(remove)
        self.grid[remove] = -1

        for j in range(self.grid_size):
            col = self.grid[:, j]
            new_col = col[col != -1]
            n_new = self.grid_size - len(new_col)
            new_col = np.concatenate((np.full(n_new, -1), new_col))
            self.grid[:, j] = new_col

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == -1:
                    self.grid[i, j] = random.randint(0, self.n_types - 1)
        return n_removed

    def step(self, action_index):
        self.steps += 1
        done = False
        reward = 0


        (i1, j1), (i2, j2) = self.actions[action_index]
        self.grid[i1, j1], self.grid[i2, j2] = self.grid[i2, j2], self.grid[i1, j1]

        remove = self._check_matches()
        if remove.any():
            reward = self._update_grid(remove)
        else:
            self.grid[i1, j1], self.grid[i2, j2] = self.grid[i2, j2], self.grid[i1, j1]
            reward = -1

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {}

#####################################
# 2. Агент: DQN и класс DQNAgent    #
#####################################

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_shape, action_size, lr=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr


        self.state_dim = state_shape[0] * state_shape[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.state_dim, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=2000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

    def act(self, state):
        state = torch.FloatTensor(state.flatten()).to(self.device)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Преобразуем список переходов в numpy-массив сразу для повышения производительности
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).astype(np.float32)).unsqueeze(1).to(self.device)

        curr_q = self.model(states).gather(1, actions)
        next_q = self.model(next_states).max(1)[0].detach().unsqueeze(1)
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(curr_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Снижаем ε (epsilon) для уменьшения вероятности случайных действий
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

###################################
# 3. Цикл обучения и визуализация #
###################################

def train():
    n_episodes = 500  # число эпизодов для обучения
    max_steps = 50    # максимальное число ходов в каждом эпизоде
    batch_size = 32

    env = Match3Env(grid_size=6, n_types=4, max_steps=max_steps)
    agent = DQNAgent(state_shape=(env.grid_size, env.grid_size), action_size=len(env.actions))
    
    best_reward = -float('inf')
    rewards_per_episode = []

    for episode in range(1, n_episodes+1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Обработка событий Pygame, чтобы окно не "не отвечало"
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Опционально: визуализируем игру во время обучения
            env.render()
            time.sleep(0.05)
            
            # Обучаем агента после каждого шага
            agent.replay(batch_size)
        
        rewards_per_episode.append(total_reward)
        print(f"Эпизод {episode}/{n_episodes}, Награда: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        
        # Сохраняем модель, если достигнута новая лучшая награда за эпизод
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), "model_best.pth")
            print("Модель обновлена и сохранена как model_best.pth!")
    
    # Отображаем график наград
    plt.plot(rewards_per_episode)
    plt.xlabel("Эпизод")
    plt.ylabel("Награда")
    plt.title("График наград по эпизодам")
    plt.show()

    # В конце сохраняем финальную модель
    torch.save(agent.model.state_dict(), "model_final.pth")
    print("Финальная модель сохранена как model_final.pth")

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pygame.quit()
        print("Обучение прервано. Pygame закрыт.")
