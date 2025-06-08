import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import math
import os
from ursina import Vec3
from config import GAME_CONFIG, PHYSICS_CONFIG, ACTIONS, DQN_CONFIG

# ----------------- DEEP Q-LEARNING SETUP -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_layer_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_layer_size)
        self.dropout1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.dropout2 = nn.Dropout(0.1)
        self.layer3 = nn.Linear(hidden_layer_size, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return self.layer3(x)

class DQNAgent:
    def __init__(self, team_name, config, state_size, action_size):
        self.team_name = team_name
        self.config = config
        self.max_reward = -float('inf') # Track the highest reward seen
        self.reward_history = deque(maxlen=500) # For calculating average reward
        self.position_history = deque(maxlen=self.config.get('STATIONARY_WINDOW', 200))
        self.noise_scale = 0.0
        self.last_loss = 0.0
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = ReplayBuffer(self.config['MEMORY_CAPACITY'])
        self.steps_done = 0

        hidden_layer_size = self.config.get('HIDDEN_LAYER_SIZE', 128)

        self.policy_net = DQN(self.state_size, self.action_size, hidden_layer_size).to(device)
        self.target_net = DQN(self.state_size, self.action_size, hidden_layer_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.perturbed_net = DQN(self.state_size, self.action_size, hidden_layer_size).to(device)
        self.perturbed_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.config['LR'], amsgrad=True)

    @property
    def average_reward(self):
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    def apply_noise(self):
        """Copies the policy network to the perturbed network and adds noise."""
        if not self.config.get('NOISE_ENABLED', False):
            self.perturbed_net.load_state_dict(self.policy_net.state_dict())
            return
            
        self.noise_scale = self.config['NOISE_SCALE_END'] + \
            (self.config['NOISE_SCALE_START'] - self.config['NOISE_SCALE_END']) * \
            math.exp(-1. * self.steps_done / self.config['NOISE_SCALE_DECAY'])

        # Load the policy network's state into the perturbed network
        self.perturbed_net.load_state_dict(self.policy_net.state_dict())
        
        # Add noise to the perturbed network's parameters
        with torch.no_grad():
            for param in self.perturbed_net.parameters():
                noise = torch.randn_like(param) * self.noise_scale
                param.add_(noise)

    def select_action(self, state):
        self.steps_done += 1
        with torch.no_grad():
            self.perturbed_net.eval() # Disable dropout for action selection
            # Action selection is now done by the perturbed network
            return self.perturbed_net(state).max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.config['BATCH_SIZE']:
            return
        
        self.policy_net.train() # Enable dropout for training

        transitions = self.memory.sample(self.config['BATCH_SIZE'])
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.config['BATCH_SIZE'], device=device)
        with torch.no_grad():
            next_state_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)

        expected_state_action_values = (next_state_values * self.config['GAMMA']) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.last_loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.config['GRAD_CLIP'])
        self.optimizer.step()

    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.config['TAU'] + target_net_state_dict[key]*(1-self.config['TAU'])
        self.target_net.load_state_dict(target_net_state_dict)

    def save_model(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, filename)
        print(f"Saving model for {self.team_name} to {path}")
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, directory, filename):
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            try:
                print(f"Loading model for {self.team_name} from {path}")
                self.policy_net.load_state_dict(torch.load(path, map_location=device))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.policy_net.eval() # Set model to evaluation mode after loading
            except RuntimeError as e:
                print(f"Could not load model for {self.team_name} due to architecture mismatch. Starting fresh.")
                print(f"Error: {e}")
        else:
            print(f"No model found for {self.team_name} at {path}, starting fresh.") 