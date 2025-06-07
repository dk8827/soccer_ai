from ursina import *
import sys
import time as a_different_time # Use a different alias to avoid conflict with ursina's time module
import torch

from config import GAME_CONFIG, DQN_CONFIG
from ai import DQNAgent, device
from game import (
    setup_field, Player, Ball,
    move_player, handle_player_collisions, handle_ball_kicks,
    calculate_base_reward
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

# These will be created/reset for each episode
agents = []
ball = None


# ----------------- AGENTS -----------------
def setup_episode():
    global agents, ball

    # Store learning progress from old agents before reset
    learning_progress = {}
    for agent in agents:
        learning_progress[agent.team_name] = {
            'policy_net_state': agent.policy_net.state_dict(),
            'target_net_state': agent.target_net.state_dict(),
            'memory': agent.memory,
            'steps_done': agent.steps_done
        }
        if agent.player:
            destroy(agent.player)

    if ball:
        destroy(ball)

    ball = Ball(position=(0,0,0))
    
    player_entities = [
        Player(position=(-15, 0, 0), color=color.orange, rotation_y=90),
        Player(position=(15, 0, 0), color=color.blue, rotation_y=-90)
    ]

    goal_assignments = [(player1_goal, player2_goal), (player2_goal, player1_goal)]
    team_names = ['player1', 'player2']
    
    agents = []
    for i in range(2):
        player = player_entities[i]
        opponent = player_entities[1-i]
        own_goal, opp_goal = goal_assignments[i]
        team_name = team_names[i]

        agent = DQNAgent(player, opponent, ball, own_goal, opp_goal, team_name, DQN_CONFIG)
        
        # Restore learning progress
        if team_name in learning_progress:
            progress = learning_progress[team_name]
            agent.policy_net.load_state_dict(progress['policy_net_state'])
            agent.target_net.load_state_dict(progress['target_net_state'])
            agent.memory = progress['memory']
            agent.steps_done = progress['steps_done']
        else: # Load from file on first ever run
            agent.load_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{team_name}.pth")

        agents.append(agent)


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
    setup_episode()
    episode_frame_count = 0
    update_score_ui()
    if GAME_CONFIG['SHOULD_RENDER']:
        update_timer_ui()
        # Workaround for a rendering issue on macOS where the window may not
        # update until user interaction. This forces a redraw.
        if sys.platform == 'darwin':
            window.size += (0, 1)
            window.size -= (0, 1)

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
    states = [agent.get_state() for agent in agents]
    actions = [agent.select_action(state) for agent, state in zip(agents, states)]

    for agent, action in zip(agents, actions):
        move_player(agent.player, action)

    handle_player_collisions(agents[0].player, agents[1].player)

    # --- REWARD CALCULATION ---
    rewards = {agent.team_name: calculate_base_reward(agent, ball) for agent in agents}

    # --- PHYSICS & KICK REWARDS ---
    prev_ball_dist_to_opp_goals = {agent.team_name: distance_xz(ball.position, agent.opp_goal.position) for agent in agents}
    hit_info = ball.intersects()
    if hit_info.hit:
        kick_rewards = handle_ball_kicks(hit_info, ball, agents, prev_ball_dist_to_opp_goals)
        for team_name, reward in kick_rewards.items():
            rewards[team_name] += reward
    
    # --- GOAL CHECK & TERMINAL REWARDS ---
    done = False
    goal_scored_by_player1 = ball.intersects(player2_goal.trigger).hit
    goal_scored_by_player2 = ball.intersects(player1_goal.trigger).hit

    if goal_scored_by_player1:
        score['player1'] += 1
        rewards['player1'] += DQN_CONFIG['REWARD_GOAL']
        rewards['player2'] += DQN_CONFIG['PENALTY_CONCEDE']
        done = True
    elif goal_scored_by_player2:
        score['player2'] += 1
        rewards['player2'] += DQN_CONFIG['REWARD_GOAL']
        rewards['player1'] += DQN_CONFIG['PENALTY_CONCEDE']
        done = True

    # --- LEARNING STEP ---
    next_states = [agent.get_state() if not done else None for agent in agents]
    for i, agent in enumerate(agents):
        reward_tensor = torch.tensor([rewards[agent.team_name]], device=device)
        agent.memory.push(states[i], actions[i], next_states[i], reward_tensor)

    TOTAL_FRAMES += 1
    episode_frame_count += 1

    if TOTAL_FRAMES % DQN_CONFIG['UPDATE_EVERY'] == 0:
        for agent in agents:
            agent.optimize_model()

    if TOTAL_FRAMES % DQN_CONFIG['TARGET_UPDATE_EVERY'] == 0:
        for agent in agents:
            agent.update_target_net()

    # --- CHECKPOINTING & EPISODE RESET ---
    if TOTAL_FRAMES > 0 and TOTAL_FRAMES % 50000 == 0:
        for agent in agents:
            agent.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{agent.team_name}_{TOTAL_FRAMES}.pth")

    if done:
        if GAME_CONFIG['SHOULD_RENDER']: update_score_ui()
        setup_episode()
        episode_frame_count = 0

# ----------------- SIMULATION RUNNER -----------------
def input(key):
    if key == 'escape':
        print("--- Simulation ended by user. Saving models... ---")
        for agent in agents:
            agent.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{agent.team_name}_final.pth")
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
    for agent in agents:
        agent.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{agent.team_name}_final.pth")
    sys.exit()

if __name__ == '__main__':
    run_simulation(GAME_CONFIG['NUM_GAMES_TO_RUN'])