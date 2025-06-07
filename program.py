from ursina import *
import sys
import math
from panda3d.core import Quat
import time as a_different_time # Use a different alias to avoid conflict with ursina's time module
import os

# --- New Imports for AI ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

# ----------------- CONFIGURATION -----------------

# --- Game & Simulation Config ---
GAME_CONFIG = {
    'SHOULD_RENDER': True,
    'FIELD_WIDTH': 40,
    'FIELD_LENGTH': 25,
    'GAME_TIMER_SECONDS': 60,
    'NUM_GAMES_TO_RUN': 100,
    'SAVE_DIR': "models",
}

# --- Physics & Gameplay Config ---
PHYSICS_CONFIG = {
    'PLAYER_MOVE_SPEED': 12,
    'PLAYER_TURN_SPEED': 200,
    'KICK_STRENGTH': 15,
    'KICK_LIFT': 4,
}

# --- AI Hyperparameters ---
DQN_CONFIG = {
    # Network and State
    'STATE_SIZE': 14,  # Updated: 12 original + 2 new angle features
    'ACTION_SIZE': 3,  # Turn Left, Turn Right, Move Forward
    # Training
    'BATCH_SIZE': 128,
    'GAMMA': 0.99,       # Discount factor
    'LR': 1e-4,          # Learning Rate
    'TAU': 0.005,        # Target network soft update rate
    'MEMORY_CAPACITY': 50000,
    'UPDATE_EVERY': 4,              # How often to run the optimization step
    'TARGET_UPDATE_EVERY': 100,     # How often to soft-update the target network
    # Epsilon-Greedy Exploration
    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'EPS_DECAY': 30000, # Slower decay for more exploration
    # Rewards
    'REWARD_GOAL': 100,
    'REWARD_KICK': 5,
    'REWARD_KICK_TOWARDS_GOAL': 10,
    'REWARD_MOVE_TO_BALL_SCALE': 0.5,
    'REWARD_DEFENSIVE_POS': 0.1,
    'PENALTY_TIME': -0.01,
    'PENALTY_CONCEDE': -100,
}


# ----------------- GAME SETUP -----------------
if not GAME_CONFIG['SHOULD_RENDER']:
    app = Ursina(headless=True)
else:
    app = Ursina()

window.title = 'AI Cube Soccer 3D'
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False
window.fps_counter.enabled = True

# Global state variables that will be reset each game
score = {'orange': 0, 'blue': 0}
game_over = False
time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
game_number = 0

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
        self.last_dist_to_ball = None

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

    def move_player(self, action):
        action_id = action.item()
        dt = time.dt
        if dt == 0: return

        if action_id == 0: # Turn Left
            self.player.rotation_y -= PHYSICS_CONFIG['PLAYER_TURN_SPEED'] * dt
        elif action_id == 1: # Turn Right
            self.player.rotation_y += PHYSICS_CONFIG['PLAYER_TURN_SPEED'] * dt
        elif action_id == 2: # Move Forward
            self.player.position += self.player.forward * dt * PHYSICS_CONFIG['PLAYER_MOVE_SPEED']

        self.player.x = clamp(self.player.x, -GAME_CONFIG['FIELD_WIDTH']/2, GAME_CONFIG['FIELD_WIDTH']/2)
        self.player.z = clamp(self.player.z, -GAME_CONFIG['FIELD_LENGTH']/2, GAME_CONFIG['FIELD_LENGTH']/2)

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

# ----------------- ENVIRONMENT -----------------
ground = Entity(model='quad', texture='assets/grid_texture.png', scale=(GAME_CONFIG['FIELD_WIDTH'], GAME_CONFIG['FIELD_LENGTH']), rotation_x=90, collider='box', y=-0.5)
# ... Walls and goal frames setup ...
wall_texture = 'assets/wall_texture.png'; wall_height = 12; goal_width = 20
wall_segment_length = (GAME_CONFIG['FIELD_LENGTH'] - goal_width) / 2
wall_z_offset = goal_width/2 + wall_segment_length/2
Entity(model='cube', texture=wall_texture, scale=(GAME_CONFIG['FIELD_WIDTH'] + 2, wall_height, 1), position=(0, wall_height/2 - 0.5, GAME_CONFIG['FIELD_LENGTH']/2 + 0.5), collider='box')
Entity(model='cube', scale=(GAME_CONFIG['FIELD_WIDTH'] + 2, wall_height, 1), position=(0, wall_height/2 - 0.5, -GAME_CONFIG['FIELD_LENGTH']/2 - 0.5), collider='box', visible=False)
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(-GAME_CONFIG['FIELD_WIDTH']/2 - 0.5, wall_height/2 - 0.5, -wall_z_offset), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(-GAME_CONFIG['FIELD_WIDTH']/2 - 0.5, wall_height/2 - 0.5, wall_z_offset), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(GAME_CONFIG['FIELD_WIDTH']/2 + 0.5, wall_height/2 - 0.5, -wall_z_offset), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(GAME_CONFIG['FIELD_WIDTH']/2 + 0.5, wall_height/2 - 0.5, wall_z_offset), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, 1.25, goal_width), position=(-GAME_CONFIG['FIELD_WIDTH']/2 - 0.5, 10.875, 0), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, 1.25, goal_width), position=(GAME_CONFIG['FIELD_WIDTH']/2 + 0.5, 10.875, 0), collider='box')

class Goal(Entity):
    def __init__(self, clr, **kwargs):
        super().__init__(**kwargs)
        w, h = 20, 10; recess_depth = 2; frame_color = color.orange if clr == 'orange' else color.blue
        Entity(parent=self, model='cube', collider='box', scale=(w, .5, .5), position=(0, h, recess_depth), color=frame_color)
        Entity(parent=self, model='cube', collider='box', scale=(.5, h, .5), position=(-w/2, h/2, recess_depth), color=frame_color)
        Entity(parent=self, model='cube', collider='box', scale=(.5, h, .5), position=(w/2, h/2, recess_depth), color=frame_color)
        Entity(parent=self, model='quad', texture='assets/net_texture.png', double_sided=True, texture_scale=(w/10, h/5), scale=(w,h), position=(0, h/2, recess_depth + 0.05))
        self.trigger = Entity(parent=self, model='cube', collider='box', scale=(w-0.5, h-0.5, 1), position=(0, h/2, recess_depth + 0.5), visible=False)

orange_goal = Goal(clr='orange', position=(-GAME_CONFIG['FIELD_WIDTH']/2 - 0.5, 0, 0), rotation_y=-90)
blue_goal = Goal(clr='blue', position=(GAME_CONFIG['FIELD_WIDTH']/2 + 0.5, 0, 0), rotation_y=90)

class Player(Entity):
    def __init__(self, **kwargs):
        super().__init__(model='cube', scale=1.8, collider='box', **kwargs)
        eye_dist = 0.25; eye_y = 0.2; eye_z_offset = 0.51
        Entity(parent=self, model='sphere', color=color.white, scale=0.3, position=(-eye_dist, eye_y, eye_z_offset))
        Entity(parent=self, model='sphere', color=color.black, scale=0.15, position=(-eye_dist, eye_y, eye_z_offset + 0.01))
        Entity(parent=self, model='sphere', color=color.white, scale=0.3, position=(eye_dist, eye_y, eye_z_offset))
        Entity(parent=self, model='sphere', color=color.black, scale=0.15, position=(eye_dist, eye_y, eye_z_offset + 0.01))

class Ball(Entity):
    def __init__(self, position):
        super().__init__(model='sphere', texture='assets/soccer_ball_texture.png', position=position, scale=1.2, collider='sphere')
        self.velocity = Vec3(0,0,0)
        self.gravity = -9.8

    def update(self):
        dt = time.dt
        if dt == 0: return

        if self.velocity.xz.length_squared() > 0:
            distance = self.velocity.xz.length() * dt
            radius = self.scale.x / 2
            rotation_amount = (distance / radius) * (180 / math.pi)
            rotation_axis = self.velocity.cross(Vec3(0, 1, 0)).normalized() # Use built-in cross product
            q = Quat(); q.setFromAxisAngle(rotation_amount, rotation_axis)
            self.quaternion = q * self.quaternion

        self.velocity.y += self.gravity * dt
        self.position += self.velocity * dt

        ground_level = ground.y + self.scale.y/2
        if self.y < ground_level:
            self.y = ground_level
            self.velocity.y *= -0.6

        if abs(self.x) > GAME_CONFIG['FIELD_WIDTH']/2:
            if abs(self.z) > goal_width/2:
                self.velocity.x *= -0.9
                self.x = math.copysign(GAME_CONFIG['FIELD_WIDTH']/2, self.x)
        if abs(self.z) > GAME_CONFIG['FIELD_LENGTH']/2:
            self.velocity.z *= -0.9
            self.z = math.copysign(GAME_CONFIG['FIELD_LENGTH']/2, self.z)

        on_ground = self.y <= ground_level + 0.01
        if on_ground:
            self.velocity.xz = lerp(self.velocity.xz, Vec2(0,0), dt * 1.0)
            if abs(self.velocity.y) < 1: self.velocity.y = 0
        else:
            self.velocity = lerp(self.velocity, Vec3(0,0,0), dt * 0.1)

# Create entities & agents
player_orange_entity = Player(position=(-15, 0, 0), color=color.orange, rotation_y=90)
player_blue_entity = Player(position=(15, 0, 0), color=color.blue, rotation_y=-90)
ball = Ball(position=(0,0,0))

agent_orange = DQNAgent(player_orange_entity, player_blue_entity, ball, orange_goal, blue_goal, 'orange', DQN_CONFIG)
agent_blue = DQNAgent(player_blue_entity, player_orange_entity, ball, blue_goal, orange_goal, 'blue', DQN_CONFIG)

agent_orange.load_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_orange.pth")
agent_blue.load_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_blue.pth")

# UI
score_display = Text(origin=(0,0), y=0.4, scale=1.5, background=True)
timer_display = Text("00:00", origin=(0, -21), scale=1.5, background=True)
def update_score_ui():
    score_display.text = f"<orange>{score['orange']}<default> - <azure>{score['blue']}"

# ----------------- GAME LOGIC & NEW HELPER FUNCTIONS -----------------
camera.position = (0, 55, -55); camera.rotation = (45, 0, 0)
episode_frame_count = 0; TOTAL_FRAMES = 0

def handle_agent_step(agent, state):
    action = agent.select_action(state)
    agent.move_player(action)
    return action

def handle_ball_kicks(hit_info, prev_ball_dist_to_orange_goal, prev_ball_dist_to_blue_goal):
    kick_reward_orange, kick_reward_blue = 0, 0
    if hit_info.entity == player_orange_entity:
        ball.velocity = player_orange_entity.forward * PHYSICS_CONFIG['KICK_STRENGTH'] + Vec3(0, PHYSICS_CONFIG['KICK_LIFT'], 0)
        kick_reward_orange += DQN_CONFIG['REWARD_KICK']
        if distance_xz(ball.position, blue_goal.position) < prev_ball_dist_to_blue_goal:
            kick_reward_orange += DQN_CONFIG['REWARD_KICK_TOWARDS_GOAL']
    elif hit_info.entity == player_blue_entity:
        ball.velocity = player_blue_entity.forward * PHYSICS_CONFIG['KICK_STRENGTH'] + Vec3(0, PHYSICS_CONFIG['KICK_LIFT'], 0)
        kick_reward_blue += DQN_CONFIG['REWARD_KICK']
        if distance_xz(ball.position, orange_goal.position) < prev_ball_dist_to_orange_goal:
             kick_reward_blue += DQN_CONFIG['REWARD_KICK_TOWARDS_GOAL']
    return kick_reward_orange, kick_reward_blue

def calculate_base_reward(agent):
    reward = DQN_CONFIG['PENALTY_TIME']
    
    # Reward for moving closer to the ball
    current_dist_to_ball = distance_xz(agent.player.position, ball.position)
    if agent.last_dist_to_ball is not None:
        reward += (agent.last_dist_to_ball - current_dist_to_ball) * DQN_CONFIG['REWARD_MOVE_TO_BALL_SCALE']
    agent.last_dist_to_ball = current_dist_to_ball

    # NEW: Defensive positioning reward
    vec_goal_to_agent = (agent.player.position - agent.own_goal.position).xz
    vec_goal_to_ball = (ball.position - agent.own_goal.position).xz
    ball_on_my_side = (agent.team_name == 'orange' and ball.x < 0) or \
                      (agent.team_name == 'blue' and ball.x > 0)
    if ball_on_my_side and vec_goal_to_agent.length() < vec_goal_to_ball.length():
        reward += DQN_CONFIG['REWARD_DEFENSIVE_POS']
        
    return reward

def reset_positions():
    global episode_frame_count
    player_orange_entity.position = (-15, 0, 0); player_orange_entity.rotation = (0, 90, 0)
    player_blue_entity.position = (15, 0, 0); player_blue_entity.rotation = (0, -90, 0)
    ball.position = (0, 0, 0); ball.velocity = Vec3(0,0,0)
    agent_orange.last_dist_to_ball = None; agent_blue.last_dist_to_ball = None
    episode_frame_count = 0

def start_new_game():
    global score, time_left, game_over
    print(f"--- Starting Game {game_number + 1}/{GAME_CONFIG['NUM_GAMES_TO_RUN']} ---")
    score = {'orange': 0, 'blue': 0}
    time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
    game_over = False
    reset_positions()
    update_score_ui()

# ----------------- MAIN UPDATE LOOP -----------------
def update():
    global time_left, game_over, score, episode_frame_count, TOTAL_FRAMES
    if game_over: return
    dt = time.dt
    if dt == 0: return

    # --- TIMER LOGIC ---
    time_left -= dt
    if time_left <= 0:
        time_left = 0
        game_over = True
    if GAME_CONFIG['SHOULD_RENDER']:
        mins, secs = divmod(time_left, 60)
        timer_display.text = f"{int(mins):02}:{int(secs):02}"

    # --- AI STEP ---
    # 1. Get current states
    state_orange = agent_orange.get_state()
    state_blue = agent_blue.get_state()

    # 2. Agents select and perform actions
    action_orange = handle_agent_step(agent_orange, state_orange)
    action_blue = handle_agent_step(agent_blue, state_blue)

    # 3. Calculate base rewards (time penalty, moving to ball, defensive pos)
    reward_orange = calculate_base_reward(agent_orange)
    reward_blue = calculate_base_reward(agent_blue)

    # 4. Handle Physics and Kick Rewards
    prev_ball_dist_to_orange_goal = distance_xz(ball.position, orange_goal.position)
    prev_ball_dist_to_blue_goal = distance_xz(ball.position, blue_goal.position)
    hit_info = ball.intersects()
    if hit_info.hit:
        kick_reward_orange, kick_reward_blue = handle_ball_kicks(hit_info, prev_ball_dist_to_orange_goal, prev_ball_dist_to_blue_goal)
        reward_orange += kick_reward_orange
        reward_blue += kick_reward_blue

    # 5. Check for Goals and assign terminal rewards
    done = False
    if ball.intersects(blue_goal.trigger).hit:
        score['orange'] += 1
        reward_orange += DQN_CONFIG['REWARD_GOAL']
        reward_blue += DQN_CONFIG['PENALTY_CONCEDE']
        done = True
    if ball.intersects(orange_goal.trigger).hit:
        score['blue'] += 1
        reward_blue += DQN_CONFIG['REWARD_GOAL']
        reward_orange += DQN_CONFIG['PENALTY_CONCEDE']
        done = True

    # 6. Learning Step
    next_state_orange = agent_orange.get_state() if not done else None
    next_state_blue = agent_blue.get_state() if not done else None
    agent_orange.memory.push(state_orange, action_orange, next_state_orange, torch.tensor([reward_orange], device=device))
    agent_blue.memory.push(state_blue, action_blue, next_state_blue, torch.tensor([reward_blue], device=device))

    TOTAL_FRAMES += 1
    if TOTAL_FRAMES % DQN_CONFIG['UPDATE_EVERY'] == 0:
        agent_orange.optimize_model()
        agent_blue.optimize_model()

    if TOTAL_FRAMES % DQN_CONFIG['TARGET_UPDATE_EVERY'] == 0:
        agent_orange.update_target_net()
        agent_blue.update_target_net()

    # --- Episode Reset & Model Saving ---
    episode_frame_count += 1
    if TOTAL_FRAMES > 0 and TOTAL_FRAMES % 50000 == 0: # Checkpoint saving
        agent_orange.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_orange_{TOTAL_FRAMES}.pth")
        agent_blue.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_blue_{TOTAL_FRAMES}.pth")

    if done:
        if GAME_CONFIG['SHOULD_RENDER']: update_score_ui()
        reset_positions()

# ----------------- SIMULATION RUNNER -----------------
def input(key):
    if key == 'escape':
        print("--- Simulation ended by user. Saving models... ---")
        agent_orange.save_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_orange_final.pth")
        agent_blue.save_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_blue_final.pth")
        sys.exit()

def run_simulation(num_games_to_play):
    global game_number, game_over
    for i in range(num_games_to_play):
        game_number = i
        start_new_game()
        while not game_over:
            app.step() # Processes one frame, calling update()

        print(f"Game {game_number + 1} Finished. Final Score: <orange>Orange {score['orange']}<default> - <azure>Blue {score['blue']}<default>")
        print("-" * 40)
        if not GAME_CONFIG['SHOULD_RENDER']:
             a_different_time.sleep(0.01) # Small delay for headless console readability

    print(f"--- Simulation of {num_games_to_play} games complete. Saving final models. ---")
    agent_orange.save_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_orange_final.pth")
    agent_blue.save_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_blue_final.pth")
    sys.exit()

if __name__ == '__main__':
    run_simulation(GAME_CONFIG['NUM_GAMES_TO_RUN'])