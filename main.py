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
score = {'player1': 0, 'player2': 0}
game_over = False
time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
game_number = 0
episode_frame_count = 0
TOTAL_FRAMES = 0


# ----------------- ENVIRONMENT & ENTITIES -----------------
player1_goal, player2_goal = setup_field()

player1_entity = Player(position=(-15, 0, 0), color=color.orange, rotation_y=90)
player2_entity = Player(position=(15, 0, 0), color=color.blue, rotation_y=-90)
ball = Ball(position=(0,0,0))

# ----------------- AGENTS -----------------
agent_player1 = DQNAgent(player1_entity, player2_entity, ball, player1_goal, player2_goal, 'player1', DQN_CONFIG)
agent_player2 = DQNAgent(player2_entity, player1_entity, ball, player2_goal, player1_goal, 'player2', DQN_CONFIG)

agent_player1.load_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_player1.pth")
agent_player2.load_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_player2.pth")

# ----------------- UI -----------------
score_display = Text(origin=(0,0), y=0.4, scale=1.5, background=True)
timer_display = Text("00:00", origin=(0,0), y=0.35, scale=1.2, background=True)
def update_score_ui():
    score_display.text = f"<orange>{score['player1']}<default> - <azure>{score['player2']}"

def update_timer_ui():
    mins, secs = divmod(time_left, 60)
    timer_display.text = f"Time: {int(mins):02}:{int(secs):02}"

# ----------------- GAME LOGIC -----------------
camera.position = (0, 55, -55); camera.rotation = (45, 0, 0)

def start_new_game():
    global score, time_left, game_over, episode_frame_count
    print(f"--- Starting Game {game_number + 1}/{GAME_CONFIG['NUM_GAMES_TO_RUN']} ---")
    score = {'player1': 0, 'player2': 0}
    time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
    game_over = False
    reset_positions(player1_entity, player2_entity, ball, agent_player1, agent_player2)
    episode_frame_count = 0
    update_score_ui()
    if GAME_CONFIG['SHOULD_RENDER']:
        update_timer_ui()

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
        update_timer_ui()

    # --- AI STEP ---
    state_player1 = agent_player1.get_state()
    state_player2 = agent_player2.get_state()

    action_player1 = agent_player1.select_action(state_player1)
    action_player2 = agent_player2.select_action(state_player2)

    move_player(agent_player1.player, action_player1)
    move_player(agent_player2.player, action_player2)

    handle_player_collisions(player1_entity, player2_entity)

    # --- REWARD CALCULATION ---
    reward_player1 = calculate_base_reward(agent_player1, ball)
    reward_player2 = calculate_base_reward(agent_player2, ball)

    # --- PHYSICS & KICK REWARDS ---
    prev_ball_dist_to_player1_goal = distance_xz(ball.position, player1_goal.position)
    prev_ball_dist_to_player2_goal = distance_xz(ball.position, player2_goal.position)
    hit_info = ball.intersects()
    if hit_info.hit:
        kick_reward_player1, kick_reward_player2 = handle_ball_kicks(hit_info, ball, player1_goal, player2_goal, player1_entity, player2_entity, prev_ball_dist_to_player1_goal, prev_ball_dist_to_player2_goal)
        reward_player1 += kick_reward_player1
        reward_player2 += kick_reward_player2

    # --- GOAL CHECK & TERMINAL REWARDS ---
    done = False
    if ball.intersects(player2_goal.trigger).hit:
        score['player1'] += 1
        reward_player1 += DQN_CONFIG['REWARD_GOAL']
        reward_player2 += DQN_CONFIG['PENALTY_CONCEDE']
        done = True
    if ball.intersects(player1_goal.trigger).hit:
        score['player2'] += 1
        reward_player2 += DQN_CONFIG['REWARD_GOAL']
        reward_player1 += DQN_CONFIG['PENALTY_CONCEDE']
        done = True

    # --- LEARNING STEP ---
    next_state_player1 = agent_player1.get_state() if not done else None
    next_state_player2 = agent_player2.get_state() if not done else None
    agent_player1.memory.push(state_player1, action_player1, next_state_player1, torch.tensor([reward_player1], device=device))
    agent_player2.memory.push(state_player2, action_player2, next_state_player2, torch.tensor([reward_player2], device=device))

    TOTAL_FRAMES += 1
    episode_frame_count += 1

    if TOTAL_FRAMES % DQN_CONFIG['UPDATE_EVERY'] == 0:
        agent_player1.optimize_model()
        agent_player2.optimize_model()

    if TOTAL_FRAMES % DQN_CONFIG['TARGET_UPDATE_EVERY'] == 0:
        agent_player1.update_target_net()
        agent_player2.update_target_net()

    # --- CHECKPOINTING & EPISODE RESET ---
    if TOTAL_FRAMES > 0 and TOTAL_FRAMES % 50000 == 0:
        agent_player1.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_player1_{TOTAL_FRAMES}.pth")
        agent_player2.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_player2_{TOTAL_FRAMES}.pth")

    if done:
        if GAME_CONFIG['SHOULD_RENDER']: update_score_ui()
        reset_positions(player1_entity, player2_entity, ball, agent_player1, agent_player2)
        episode_frame_count = 0

# ----------------- SIMULATION RUNNER -----------------
def input(key):
    if key == 'escape':
        print("--- Simulation ended by user. Saving models... ---")
        agent_player1.save_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_player1_final.pth")
        agent_player2.save_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_player2_final.pth")
        sys.exit()

def run_simulation(num_games_to_play):
    global game_number, game_over
    for i in range(num_games_to_play):
        game_number = i
        start_new_game()
        while not game_over:
            app.step()

        print(f"Game {game_number + 1} Finished. Final Score: Player1 {score['player1']} - Player2 {score['player2']}")
        print("-" * 40)
        if not GAME_CONFIG['SHOULD_RENDER']:
             a_different_time.sleep(0.01)

    print(f"--- Simulation of {num_games_to_play} games complete. Saving final models. ---")
    agent_player1.save_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_player1_final.pth")
    agent_player2.save_model(GAME_CONFIG['SAVE_DIR'], "dqn_soccer_player2_final.pth")
    sys.exit()

if __name__ == '__main__':
    run_simulation(GAME_CONFIG['NUM_GAMES_TO_RUN'])