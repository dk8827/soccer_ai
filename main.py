from ursina import *
import sys
import time as a_different_time # Use a different alias to avoid conflict with ursina's time module
import torch

from config import GAME_CONFIG, DQN_CONFIG
from ai import DQNAgent, device
from game import (
    setup_field, Player, Ball,
    move_player, handle_player_collisions, handle_ball_kicks,
    calculate_base_reward, reset_positions
)


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
episode_frame_count = 0
TOTAL_FRAMES = 0


# ----------------- ENVIRONMENT & ENTITIES -----------------
orange_goal, blue_goal = setup_field()

player_orange_entity = Player(position=(-15, 0, 0), color=color.orange, rotation_y=90)
player_blue_entity = Player(position=(15, 0, 0), color=color.blue, rotation_y=-90)
ball = Ball(position=(0,0,0))

# ----------------- AGENTS -----------------
agent_orange = DQNAgent(player_orange_entity, player_blue_entity, ball, orange_goal, blue_goal, 'orange', DQN_CONFIG)
agent_blue = DQNAgent(player_blue_entity, player_orange_entity, ball, blue_goal, orange_goal, 'blue', DQN_CONFIG)

agent_orange.load_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_orange.pth")
agent_blue.load_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_blue.pth")

# ----------------- UI -----------------
score_display = Text(origin=(0,0), y=0.4, scale=1.5, background=True)
timer_display = Text("00:00", origin=(0, -21), scale=1.5, background=True)
def update_score_ui():
    score_display.text = f"<orange>{score['orange']}<default> - <azure>{score['blue']}"

# ----------------- GAME LOGIC -----------------
camera.position = (0, 55, -55); camera.rotation = (45, 0, 0)

def start_new_game():
    global score, time_left, game_over, episode_frame_count
    print(f"--- Starting Game {game_number + 1}/{GAME_CONFIG['NUM_GAMES_TO_RUN']} ---")
    score = {'orange': 0, 'blue': 0}
    time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
    game_over = False
    reset_positions(player_orange_entity, player_blue_entity, ball, agent_orange, agent_blue)
    episode_frame_count = 0
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
    state_orange = agent_orange.get_state()
    state_blue = agent_blue.get_state()

    action_orange = agent_orange.select_action(state_orange)
    action_blue = agent_blue.select_action(state_blue)

    move_player(agent_orange.player, action_orange)
    move_player(agent_blue.player, action_blue)

    handle_player_collisions(player_orange_entity, player_blue_entity)

    # --- REWARD CALCULATION ---
    reward_orange = calculate_base_reward(agent_orange, ball)
    reward_blue = calculate_base_reward(agent_blue, ball)

    # --- PHYSICS & KICK REWARDS ---
    prev_ball_dist_to_orange_goal = distance_xz(ball.position, orange_goal.position)
    prev_ball_dist_to_blue_goal = distance_xz(ball.position, blue_goal.position)
    hit_info = ball.intersects()
    if hit_info.hit:
        kick_reward_orange, kick_reward_blue = handle_ball_kicks(hit_info, ball, orange_goal, blue_goal, player_orange_entity, player_blue_entity, prev_ball_dist_to_orange_goal, prev_ball_dist_to_blue_goal)
        reward_orange += kick_reward_orange
        reward_blue += kick_reward_blue

    # --- GOAL CHECK & TERMINAL REWARDS ---
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

    # --- LEARNING STEP ---
    next_state_orange = agent_orange.get_state() if not done else None
    next_state_blue = agent_blue.get_state() if not done else None
    agent_orange.memory.push(state_orange, action_orange, next_state_orange, torch.tensor([reward_orange], device=device))
    agent_blue.memory.push(state_blue, action_blue, next_state_blue, torch.tensor([reward_blue], device=device))

    TOTAL_FRAMES += 1
    episode_frame_count += 1

    if TOTAL_FRAMES % DQN_CONFIG['UPDATE_EVERY'] == 0:
        agent_orange.optimize_model()
        agent_blue.optimize_model()

    if TOTAL_FRAMES % DQN_CONFIG['TARGET_UPDATE_EVERY'] == 0:
        agent_orange.update_target_net()
        agent_blue.update_target_net()

    # --- CHECKPOINTING & EPISODE RESET ---
    if TOTAL_FRAMES > 0 and TOTAL_FRAMES % 50000 == 0:
        agent_orange.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_orange_{TOTAL_FRAMES}.pth")
        agent_blue.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_blue_{TOTAL_FRAMES}.pth")

    if done:
        if GAME_CONFIG['SHOULD_RENDER']: update_score_ui()
        reset_positions(player_orange_entity, player_blue_entity, ball, agent_orange, agent_blue)
        episode_frame_count = 0

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
            app.step()

        print(f"Game {game_number + 1} Finished. Final Score: <orange>Orange {score['orange']}<default> - <azure>Blue {score['blue']}<default>")
        print("-" * 40)
        if not GAME_CONFIG['SHOULD_RENDER']:
             a_different_time.sleep(0.01)

    print(f"--- Simulation of {num_games_to_play} games complete. Saving final models. ---")
    agent_orange.save_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_orange_final.pth")
    agent_blue.save_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_blue_final.pth")
    sys.exit()

if __name__ == '__main__':
    run_simulation(GAME_CONFIG['NUM_GAMES_TO_RUN'])