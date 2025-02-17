# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import sys

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the neural network model
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

import torch

def save_model(agent, filename):
    # Save the model's state_dict (weights) for both the local and target Q networks
    torch.save({
        'state_dict_local': agent.qnetwork_local.state_dict(),
        'state_dict_target': agent.qnetwork_target.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'seed': agent.seed,
        'state_size': agent.state_size,
        'action_size': agent.action_size
    }, filename)
    
def load_model(filename, state_size, action_size, seed, lr):
    # Initialize a new DQN agent
    agent = DQNAgent(state_size, action_size, seed, lr)
    
    # Load the saved model's state_dicts into the agent
    checkpoint = torch.load(filename)
    agent.qnetwork_local.load_state_dict(checkpoint['state_dict_local'])
    agent.qnetwork_target.load_state_dict(checkpoint['state_dict_target'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Optionally, return additional info if needed
    agent.seed = checkpoint['seed']
    agent.state_size = checkpoint['state_size']
    agent.action_size = checkpoint['action_size']
    
    return agent
# Define the DQN agent class
class DQNAgent:
    # Initialize the DQN agent
    def __init__(self, state_size, action_size, seed, lr):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)

        self.memory = ReplayBuffer(action_size, buffer_size=int(1e5), batch_size=64, seed=seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > 64:
                experiences = self.memory.sample()
                self.learn(experiences, gamma=0.99)

    # Choose an action based on the current state
    def act(self, state, eps=0.):
        #print(type(state))
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            return action_values.argmax(dim=1).item()
        else:
            return np.random.randint(self.action_size)

    # Learn from batch of experiences
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# Initialize the environment and the agent
import gymnasium as gym
from collections import deque
import random
from custom_cartpole_v4 import CustomCartPoleEnv

gym.register(
    id="CustomCartPole-v4",
    entry_point=CustomCartPoleEnv,
)
# Set up the environment
env = gym.make("CustomCartPole-v4")

new_agent = load_model(sys.argv[1], 4, 5, 0, 0)

# Visualize the agent's performance
import pygame
import time

# Initialize pygame
pygame.init()

# Define key mappings for arrow keys
KEY_MAPPING = {
    pygame.K_LEFT: 0,   # Left arrow key (action 0)
    pygame.K_RIGHT: 4,  # Right arrow key (action 1)
}

#close old env
env.close()

env = gym.make("CustomCartPole-v4", render_mode="human")

while True:
    state = env.reset()[0]
    done = False
    
    while not done:
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()    
                pygame.quit()
                sys.exit()
        keys = pygame.key.get_pressed()
    
        if keys[pygame.K_LEFT]:
            action = KEY_MAPPING[pygame.K_LEFT]  # Left arrow key overrides agent's action
        elif keys[pygame.K_RIGHT]:
            action = KEY_MAPPING[pygame.K_RIGHT]  # Right arrow key overrides agent's action
        else:
            # Use the agent's action when no key is pressed
            action = new_agent.act(state, eps=0.)
        
        env.render()
        
    
        nextStep = env.step(action)
        #nextStep = env.step(2)
        next_state = nextStep[0]
        reward = nextStep[1]
        done = nextStep[2]
    
        state = next_state
        #time.sleep(0.1)  Add a delay to make the visualization easier to follow
    
env.close()    
pygame.quit()