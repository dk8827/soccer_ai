from ursina import *
import sys
import math
from panda3d.core import Quat
import time as a_different_time # Use a different alias to avoid conflict with ursina's time module

# --- New Imports for AI ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os

def cross(v1, v2):
    return Vec3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    )

# ----------------- GAME SETUP -----------------
# We set render mode to False to speed up training significantly.
# Set to True if you want to watch the games.
SHOULD_RENDER = True
if not SHOULD_RENDER:
    app = Ursina(headless=True)
else:
    app = Ursina()


window.title = 'AI Cube Soccer 3D'
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False
window.fps_counter.enabled = True

FIELD_WIDTH = 40
FIELD_LENGTH = 25
# Global state variables that will be reset each game
score = {'orange': 0, 'blue': 0}
game_over = False
# MODIFIED: Shortened game time for faster simulation of 100 games.
GAME_TIMER_SECONDS = 60
time_left = GAME_TIMER_SECONDS
game_number = 0
NUM_GAMES = 100

# ----------------- DEEP Q-LEARNING SETUP -----------------

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# A single transition in our environment.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    """Stores transitions to be sampled for training."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """The Neural Network for approximating Q-values."""
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
    """The AI Agent that controls a player."""
    def __init__(self, player_entity, opponent_entity, ball_entity, own_goal, opp_goal, team_name):
        # Game entities
        self.player = player_entity
        self.opponent = opponent_entity
        self.ball = ball_entity
        self.own_goal = own_goal
        self.opp_goal = opp_goal
        self.team_name = team_name

        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99  # Discount factor
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 20000 # Higher number means slower decay
        self.TAU = 0.005 # Target network update rate
        self.LR = 1e-4 # Learning Rate
        self.MEMORY_CAPACITY = 50000

        # AI state
        self.state_size = 12 # The number of inputs to the network
        self.action_size = 3 # Turn Left, Turn Right, Move Forward
        self.steps_done = 0
        # NEW: Store last distance to ball for reward shaping
        self.last_dist_to_ball = None

        # DQN setup
        self.policy_net = DQN(self.state_size, self.action_size).to(device)
        self.target_net = DQN(self.state_size, self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayBuffer(self.MEMORY_CAPACITY)

    def get_state(self):
        """Constructs the state vector for the neural network."""
        # Normalize positions by field dimensions for consistency
        norm_w = FIELD_WIDTH / 2
        norm_l = FIELD_LENGTH / 2

        p_pos = self.player.position

        # Relative vectors, normalized
        vec_to_ball = (self.ball.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_opp_goal = (self.opp_goal.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_own_goal = (self.own_goal.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_opponent = (self.opponent.position - p_pos) / Vec3(norm_w, 1, norm_l)

        # Ball velocity (normalized by a reasonable max speed)
        ball_vel = self.ball.velocity / 15.0

        # Player's forward direction
        p_fwd = self.player.forward

        state = [
            vec_to_ball.x, vec_to_ball.z,
            ball_vel.x, ball_vel.z,
            vec_to_opp_goal.x, vec_to_opp_goal.z,
            vec_to_own_goal.x, vec_to_own_goal.z,
            vec_to_opponent.x, vec_to_opponent.z,
            p_fwd.x, p_fwd.z
        ]
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    def select_action(self, state):
        """Chooses an action using epsilon-greedy policy."""
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def optimize_model(self):
        """Performs one step of the optimization (on the policy network)."""
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """Soft update of the target network's weights."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def move_player(self, action):
        """Applies the chosen action to the player entity."""
        action_id = action.item()
        turn_speed = 180
        move_speed = 10
        dt = time.dt
        if dt == 0: return # Avoid movement on the first frame of a step

        if action_id == 0: # Turn Left
            self.player.rotation_y -= turn_speed * dt
        elif action_id == 1: # Turn Right
            self.player.rotation_y += turn_speed * dt
        elif action_id == 2: # Move Forward
            self.player.position += self.player.forward * dt * move_speed

        # Clamp player position
        self.player.x = clamp(self.player.x, -FIELD_WIDTH/2, FIELD_WIDTH/2)
        self.player.z = clamp(self.player.z, -FIELD_LENGTH/2, FIELD_LENGTH/2)

    def save_model(self, path):
        """Saves the policy network weights."""
        print(f"Saving model for {self.team_name} to {path}")
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """Loads weights into both policy and target networks."""
        if os.path.exists(path):
            print(f"Loading model for {self.team_name} from {path}")
            self.policy_net.load_state_dict(torch.load(path, map_location=device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            print(f"No model found for {self.team_name} at {path}, starting fresh.")

# ----------------- ENVIRONMENT -----------------
ground = Entity(model='quad', texture='assets/grid_texture.png', scale=(FIELD_WIDTH, FIELD_LENGTH), rotation_x=90, collider='box', y=-0.5)
# ... Walls and goal frames setup (unchanged)
wall_texture = 'assets/wall_texture.png'
wall_height = 12
goal_width = 20
wall_segment_length = (FIELD_LENGTH - goal_width) / 2
wall_z_offset = goal_width/2 + wall_segment_length/2
Entity(model='cube', texture=wall_texture, scale=(FIELD_WIDTH + 2, wall_height, 1), position=(0, wall_height/2 - 0.5, FIELD_LENGTH/2 + 0.5), collider='box')
Entity(model='cube', scale=(FIELD_WIDTH + 2, wall_height, 1), position=(0, wall_height/2 - 0.5, -FIELD_LENGTH/2 - 0.5), collider='box', visible=False)
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(-FIELD_WIDTH/2 - 0.5, wall_height/2 - 0.5, -wall_z_offset), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(-FIELD_WIDTH/2 - 0.5, wall_height/2 - 0.5, wall_z_offset), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(FIELD_WIDTH/2 + 0.5, wall_height/2 - 0.5, -wall_z_offset), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(FIELD_WIDTH/2 + 0.5, wall_height/2 - 0.5, wall_z_offset), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, 1.25, goal_width), position=(-FIELD_WIDTH/2 - 0.5, 10.875, 0), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, 1.25, goal_width), position=(FIELD_WIDTH/2 + 0.5, 10.875, 0), collider='box')

class Goal(Entity):
    def __init__(self, clr, **kwargs):
        super().__init__(**kwargs)
        w, h = 20, 10
        recess_depth = 2
        frame_color = color.orange if clr == 'orange' else color.blue
        Entity(parent=self, model='cube', collider='box', scale=(w, .5, .5), position=(0, h, recess_depth), color=frame_color)
        Entity(parent=self, model='cube', collider='box', scale=(.5, h, .5), position=(-w/2, h/2, recess_depth), color=frame_color)
        Entity(parent=self, model='cube', collider='box', scale=(.5, h, .5), position=(w/2, h/2, recess_depth), color=frame_color)
        Entity(parent=self, model='quad', texture='assets/net_texture.png', double_sided=True, texture_scale=(w/10, h/5), scale=(w,h), position=(0, h/2, recess_depth + 0.05))
        self.trigger = Entity(parent=self, model='cube', collider='box', scale=(w-0.5, h-0.5, 1), position=(0, h/2, recess_depth + 0.5), visible=False)

orange_goal = Goal(clr='orange', position=(-FIELD_WIDTH/2 - 0.5, 0, 0), rotation_y=-90)
blue_goal = Goal(clr='blue', position=(FIELD_WIDTH/2 + 0.5, 0, 0), rotation_y=90)


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
            rotation_axis = cross(self.velocity, Vec3(0, 1, 0)).normalized()
            q = Quat()
            q.setFromAxisAngle(rotation_amount, rotation_axis)
            self.quaternion = q * self.quaternion

        self.velocity.y += self.gravity * dt
        self.position += self.velocity * dt

        ground_level = ground.y + self.scale.y/2
        if self.y < ground_level:
            self.y = ground_level
            self.velocity.y *= -0.6

        if abs(self.x) > FIELD_WIDTH/2:
            if abs(self.z) > goal_width/2:
                self.velocity.x *= -0.9
                self.x = math.copysign(FIELD_WIDTH/2, self.x)
        if abs(self.z) > FIELD_LENGTH/2:
            self.velocity.z *= -0.9
            self.z = math.copysign(FIELD_LENGTH/2, self.z)

        on_ground = self.y <= ground_level + 0.01
        if on_ground:
            self.velocity.xz = lerp(self.velocity.xz, Vec2(0,0), dt * 1.0)
            if abs(self.velocity.y) < 1: self.velocity.y = 0
        else:
            self.velocity = lerp(self.velocity, Vec3(0,0,0), dt * 0.1)

# Create the entities
player_orange_entity = Player(position=(-10, 0, 0), color=color.orange, rotation_y=90)
player_blue_entity = Player(position=(10, 0, 0), color=color.blue, rotation_y=-90)
ball = Ball(position=(0,0,0))

# Create the AI agents
agent_orange = DQNAgent(player_orange_entity, player_blue_entity, ball, orange_goal, blue_goal, 'orange')
agent_blue = DQNAgent(player_blue_entity, player_orange_entity, ball, blue_goal, orange_goal, 'blue')

# Load pre-trained models if they exist
agent_orange.load_model("dqn_soccer_orange.pth")
agent_blue.load_model("dqn_soccer_blue.pth")

# ----------------- UI -----------------
score_display = Text(origin=(0,0), y=0.4, scale=1.5, background=True)
timer_display = Text("00:00", origin=(0, -21), scale=1.5, background=True)
game_over_text = Text("", origin=(0,0), scale=3, background=True) # Not used, but can be
def update_score_ui():
    score_display.text = f"<orange>{score['orange']}<default> - <azure>{score['blue']}"

# ----------------- CAMERA AND GAME LOGIC -----------------
camera.position = (0, 55, -55); camera.rotation = (45, 0, 0)
episode_frame_count = 0
TOTAL_FRAMES = 0
SAVE_INTERVAL = 50000

def input(key):
    if key == 'escape':
        print("--- Simulation ended by user. Saving models... ---")
        agent_orange.save_model("dqn_soccer_orange.pth")
        agent_blue.save_model("dqn_soccer_blue.pth")
        sys.exit()

def reset_positions():
    """Resets players and ball to starting positions for a new episode (after a goal)."""
    player_orange_entity.position = (-10, 0, 0); player_orange_entity.rotation = (0, 90, 0)
    player_blue_entity.position = (10, 0, 0); player_blue_entity.rotation = (0, -90, 0)
    ball.position = (0, 0, 0)
    ball.velocity = Vec3(0,0,0)

    # Reset agent's memory of last distance for fair reward calculation
    agent_orange.last_dist_to_ball = None
    agent_blue.last_dist_to_ball = None

    global episode_frame_count
    episode_frame_count = 0

def start_new_game():
    """Resets the entire game state for a new match."""
    global score, time_left, game_over
    print(f"--- Starting Game {game_number + 1}/{NUM_GAMES} ---")
    score = {'orange': 0, 'blue': 0}
    time_left = GAME_TIMER_SECONDS
    game_over = False
    reset_positions()
    update_score_ui()

def update():
    global time_left, game_over, score, episode_frame_count, TOTAL_FRAMES
    if game_over: return

    dt = time.dt
    if dt == 0: return

    # --- TIMER LOGIC ---
    time_left -= dt
    if time_left <= 0:
        time_left = 0
        game_over = True # End the game, the main loop will handle the rest
    
    if SHOULD_RENDER:
        mins, secs = divmod(time_left, 60)
        timer_display.text = f"{int(mins):02}:{int(secs):02}"

    # --- AI LOGIC: State, Action, Reward, Learn ---
    state_orange = agent_orange.get_state()
    state_blue = agent_blue.get_state()

    action_orange = agent_orange.select_action(state_orange)
    action_blue = agent_blue.select_action(state_blue)
    agent_orange.move_player(action_orange)
    agent_blue.move_player(action_blue)

    # --- REWARD CALCULATION ---
    # Start with a small penalty for each frame to encourage fast goals
    reward_orange = -0.01
    reward_blue = -0.01

    # INCENTIVE: Reward agents for getting closer to the ball
    current_dist_orange_to_ball = distance_xz(player_orange_entity.position, ball.position)
    if agent_orange.last_dist_to_ball is not None:
        reward_orange += (agent_orange.last_dist_to_ball - current_dist_orange_to_ball)
    agent_orange.last_dist_to_ball = current_dist_orange_to_ball

    current_dist_blue_to_ball = distance_xz(player_blue_entity.position, ball.position)
    if agent_blue.last_dist_to_ball is not None:
        reward_blue += (agent_blue.last_dist_to_ball - current_dist_blue_to_ball)
    agent_blue.last_dist_to_ball = current_dist_blue_to_ball

    # Store ball distances to goals *before* collision for reward calculation
    prev_ball_dist_to_blue_goal = distance_xz(ball.position, blue_goal.position)
    prev_ball_dist_to_orange_goal = distance_xz(ball.position, orange_goal.position)

    # --- COLLISION LOGIC ---
    hit_info = ball.intersects()
    if hit_info.hit:
        kick_strength = 15
        kick_lift = 4
        if hit_info.entity == player_orange_entity:
            ball.velocity = player_orange_entity.forward * kick_strength + Vec3(0, kick_lift, 0)
            reward_orange += 5 # Reward for touching the ball
            # Bigger reward for kicking towards opponent's goal
            new_ball_dist_to_blue_goal = distance_xz(ball.position, blue_goal.position)
            if new_ball_dist_to_blue_goal < prev_ball_dist_to_blue_goal:
                reward_orange += 10

        elif hit_info.entity == player_blue_entity:
            ball.velocity = player_blue_entity.forward * kick_strength + Vec3(0, kick_lift, 0)
            reward_blue += 5
            # Bigger reward for kicking towards opponent's goal
            new_ball_dist_to_orange_goal = distance_xz(ball.position, orange_goal.position)
            if new_ball_dist_to_orange_goal < prev_ball_dist_to_orange_goal: # FIXED: Compare to previous ball distance
                 reward_blue += 10

    # --- GOAL LOGIC (ends an episode, but not the game) ---
    done = False
    if ball.intersects(blue_goal.trigger).hit:
        score['orange'] += 1
        reward_orange += 100
        reward_blue -= 100
        done = True

    if ball.intersects(orange_goal.trigger).hit:
        score['blue'] += 1
        reward_blue += 100
        reward_orange -= 100
        done = True

    # --- LEARNING STEP ---
    next_state_orange = agent_orange.get_state() if not done else None
    next_state_blue = agent_blue.get_state() if not done else None

    agent_orange.memory.push(state_orange, action_orange, next_state_orange, torch.tensor([reward_orange], device=device))
    agent_blue.memory.push(state_blue, action_blue, next_state_blue, torch.tensor([reward_blue], device=device))

    agent_orange.optimize_model()
    agent_blue.optimize_model()

    agent_orange.update_target_net()
    agent_blue.update_target_net()

    # --- COUNTERS ---
    episode_frame_count += 1
    TOTAL_FRAMES += 1
    if TOTAL_FRAMES > 0 and TOTAL_FRAMES % SAVE_INTERVAL == 0:
        agent_orange.save_model(f"dqn_soccer_orange_{TOTAL_FRAMES}.pth")
        agent_blue.save_model(f"dqn_soccer_blue_{TOTAL_FRAMES}.pth")

    if done:
        if SHOULD_RENDER:
            update_score_ui()
        reset_positions()

def run_simulation(num_games_to_play):
    """
    Main loop to run multiple games, handle resets, and print stats.
    """
    global game_number, game_over
    for i in range(num_games_to_play):
        game_number = i
        start_new_game()

        # This loop runs one full game until the timer runs out
        while not game_over:
            # The app.step() function processes one frame, including calling our update() function
            app.step()

        # Game has finished, print the statistics
        print(f"Game {game_number + 1} Finished. Final Score: <orange>Orange {score['orange']}<default> - <azure>Blue {score['blue']}<default>")
        print("-" * 40)
        # Add a small delay to make console output readable between games
        a_different_time.sleep(0.1)

    print("--- Simulation of 100 games complete. Saving final models. ---")
    agent_orange.save_model("dqn_soccer_orange_final.pth")
    agent_blue.save_model("dqn_soccer_blue_final.pth")
    sys.exit()

# This is now the main entry point of the script
if __name__ == '__main__':
    run_simulation(NUM_GAMES)