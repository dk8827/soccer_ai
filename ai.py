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
from config import GAME_CONFIG, PHYSICS_CONFIG

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
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, player_entity, opponent_entity, ball_entity, own_goal, opp_goal, team_name, config):
        self.player = player_entity
        self.opponent = opponent_entity
        self.ball = ball_entity
        self.own_goal = own_goal
        self.opp_goal = opp_goal
        self.team_name = team_name
        self.config = config

        self.state_size = self.config['STATE_SIZE']
        self.action_size = self.config['ACTION_SIZE']
        self.memory = ReplayBuffer(self.config['MEMORY_CAPACITY'])
        self.steps_done = 0

        self.policy_net = DQN(self.state_size, self.action_size).to(device)
        self.target_net = DQN(self.state_size, self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.config['LR'], amsgrad=True)

    def get_state(self):
        norm_w = GAME_CONFIG['FIELD_WIDTH'] / 2
        norm_l = GAME_CONFIG['FIELD_LENGTH'] / 2
        p_pos = self.player.position

        # Relative vectors, normalized
        vec_to_ball = (self.ball.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_opp_goal = (self.opp_goal.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_own_goal = (self.own_goal.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_opponent = (self.opponent.position - p_pos) / Vec3(norm_w, 1, norm_l)

        ball_vel = self.ball.velocity / PHYSICS_CONFIG['KICK_STRENGTH']
        p_fwd = self.player.forward

        # NEW: Dot products to represent angles
        angle_to_ball = p_fwd.dot(vec_to_ball.normalized())
        angle_to_opp_goal = p_fwd.dot(vec_to_opp_goal.normalized())

        state = [
            vec_to_ball.x, vec_to_ball.z,
            ball_vel.x, ball_vel.z,
            vec_to_opp_goal.x, vec_to_opp_goal.z,
            vec_to_own_goal.x, vec_to_own_goal.z,
            vec_to_opponent.x, vec_to_opponent.z,
            p_fwd.x, p_fwd.z,
            angle_to_ball,
            angle_to_opp_goal
        ]
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.config['EPS_END'] + (self.config['EPS_START'] - self.config['EPS_END']) * \
            math.exp(-1. * self.steps_done / self.config['EPS_DECAY'])
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.config['BATCH_SIZE']:
            return

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
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.config['GAMMA']) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
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
            print(f"Loading model for {self.team_name} from {path}")
            self.policy_net.load_state_dict(torch.load(path, map_location=device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            print(f"No model found for {self.team_name} at {path}, starting fresh.") 