from ursina import *
import sys
import time as a_different_time # Use a different alias to avoid conflict with ursina's time module
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from config import GAME_CONFIG
from ui import UIManager
from game import (
    setup_field,
    move_player, handle_player_collisions,
    apply_kick_force, calculate_rewards, RewardContext,
    distance_xz
)
from managers import EntityManager, AgentManager


# ----------------- URSINA APP SETUP -----------------
if not GAME_CONFIG['SHOULD_RENDER']:
    app = Ursina(headless=True)
else:
    app = Ursina()

window.title = 'AI Cube Soccer 3D'
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False
window.fps_counter.enabled = True
camera.position = (0, 55, -55); camera.rotation = (45, 0, 0)


class GameManager:
    """
    Manages the overall game flow, state, and coordination between
    the entity and agent managers.
    """
    def __init__(self):
        # Game State
        self.score = {'player1': 0, 'player2': 0}
        self.total_score = {'player1': 0, 'player2': 0}
        self.game_over = False
        self.simulation_running = False
        self.waiting_for_reset = False
        self.time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
        self.no_touch_timer = 0
        self.game_number = 0
        self.episode_frame_count = 0
        self.TOTAL_FRAMES = 0
        self.plot_history = {
            'games': [],
            'player1_avg_reward': [], 'player2_avg_reward': [],
            'player1_loss': [], 'player2_loss': [],
            'player1_noise': [], 'player2_noise': [],
        }

        # UI & Environment
        self.ui_manager = UIManager(start_callback=self.start_simulation)
        player1_goal, player2_goal, ground = setup_field()
        
        # Managers
        self.entity_manager = EntityManager(ground, player1_goal, player2_goal)
        self.agent_manager = AgentManager(self.entity_manager, player1_goal, player2_goal)


    def start_simulation(self):
        """Callback to start the simulation, typically from a UI button."""
        if self.simulation_running: return
        self.simulation_running = True
        if GAME_CONFIG['SHOULD_RENDER']:
            self.ui_manager.destroy_start_button()
        self.start_new_game()


    def start_new_game(self):
        """Initializes a new game, resetting scores and timers."""
        print(f"--- Starting Game {self.game_number + 1}/{GAME_CONFIG['NUM_GAMES_TO_RUN']} ---")
        self.score = {'player1': 0, 'player2': 0}
        self.time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
        self.no_touch_timer = 0
        self.game_over = False
        self.waiting_for_reset = False
        self.episode_frame_count = 0
        self.ui_manager.update_score(self.total_score)
        self.ui_manager.update_timer(self.time_left)
        
        self.entity_manager.reset_episode(self.TOTAL_FRAMES)
        self.agent_manager.reset_episode()

        if GAME_CONFIG['SHOULD_RENDER'] and sys.platform == 'darwin':
            window.size += (0, 1); window.size -= (0, 1)


    def update(self):
        """Main update loop, called by Ursina every frame."""
        if not self.simulation_running or self.waiting_for_reset:
            return

        if self.game_over:
            self._log_and_plot_progress()
            print(f"Game {self.game_number + 1} Finished. Final Score: Player1 {self.score['player1']} - Player2 {self.score['player2']}")
            print("-" * 40)
            self.game_number += 1
            if self.game_number < GAME_CONFIG['NUM_GAMES_TO_RUN']:
                self.start_new_game()
            else:
                self.end_simulation()
            return

        dt = time.dt
        if dt == 0: return

        self._update_timer(dt)
        if self.game_over: return # Check again as timer can end game
        
        # Inactivity timer check
        self.no_touch_timer += dt
        if self.no_touch_timer > GAME_CONFIG['NO_TOUCH_TIMEOUT']:
            print("--- Game ended due to inactivity. ---")
            self.game_over = True
            return

        # 1. Get AI actions and move players
        states, actions = self.agent_manager.get_actions_and_states()
        for i, action in enumerate(actions):
            move_player(self.entity_manager.players[i], action)
        handle_player_collisions(self.entity_manager.players[0], self.entity_manager.players[1])

        # 2. Handle ball physics and interactions
        ball = self.entity_manager.ball
        hit_info = ball.intersects(self.entity_manager.players[0]) or ball.intersects(self.entity_manager.players[1])
        
        if hit_info.hit:
            self.no_touch_timer = 0 # Reset inactivity timer
            
            # Find the agent that hit the ball and reset its timer
            for agent in self.agent_manager.agents:
                if agent.player == hit_info.entity:
                    break # Assuming one player hits at a time

            apply_kick_force(hit_info, ball, self.agent_manager.agents)

        # 3. Calculate rewards
        ctx = RewardContext(
            agents=self.agent_manager.agents,
            ball=ball,
            player1_goal=self.entity_manager.player1_goal,
            player2_goal=self.entity_manager.player2_goal,
            hit_info=hit_info,
        )
        rewards, done, scoring_team = calculate_rewards(ctx)

        # 4. Update AI learning
        self.TOTAL_FRAMES += 1
        self.episode_frame_count += 1
        self.agent_manager.update_learning(states, actions, rewards, done, self.TOTAL_FRAMES)
        self.ui_manager.update_reward_displays(self.agent_manager.agents)
        self.ui_manager.update_game_info(self.game_number, self.TOTAL_FRAMES)

        # 5. Check for episode end
        if done:
            if scoring_team:
                self.waiting_for_reset = True
                self.score[scoring_team] += 1
                self.total_score[scoring_team] += 1
                self.ui_manager.update_score(self.total_score)
                
                # Use a callback to reset the episode after the "Goal!" message
                def on_goal_anim_complete():
                    self.entity_manager.reset_episode(self.TOTAL_FRAMES)
                    self.agent_manager.reset_episode()
                    self.episode_frame_count = 0
                    self.waiting_for_reset = False

                self.ui_manager.flash_goal_scored(scoring_team, on_goal_anim_complete)
            else:
                 # If no one scored (e.g., timeout), reset immediately
                self.entity_manager.reset_episode(self.TOTAL_FRAMES)
                self.agent_manager.reset_episode()
                self.episode_frame_count = 0


    def _update_timer(self, dt):
        """Handles the game timer."""
        self.time_left -= dt
        if self.time_left <= 0:
            self.time_left = 0
            self.game_over = True
        self.ui_manager.update_timer(self.time_left)
        self.ui_manager.update_no_touch_timer(self.no_touch_timer)


    def _log_and_plot_progress(self):
        """Logs training stats to the console and saves a progress plot."""
        if self.game_number == 0 and self.TOTAL_FRAMES < 100: # Don't log for a non-started game
            return

        agent1 = self.agent_manager.agents[0]
        agent2 = self.agent_manager.agents[1]

        # 1. Log to console
        print("\n" + "-" * 15 + f" Training Summary for Game {self.game_number + 1} " + "-" * 15)
        print(f"  Player 1: Avg Reward: {agent1.average_reward:<8.2f} | Last Loss: {agent1.last_loss:<8.4f} | Noise: {agent1.noise_scale:<8.4f}")
        print(f"  Player 2: Avg Reward: {agent2.average_reward:<8.2f} | Last Loss: {agent2.last_loss:<8.4f} | Noise: {agent2.noise_scale:<8.4f}")
        print("-" * 59 + "\n")

        # 2. Update history for plotting
        history = self.plot_history
        history['games'].append(self.game_number + 1)
        history['player1_avg_reward'].append(agent1.average_reward)
        history['player2_avg_reward'].append(agent2.average_reward)
        history['player1_loss'].append(agent1.last_loss if agent1.last_loss != 0 else None)
        history['player2_loss'].append(agent2.last_loss if agent2.last_loss != 0 else None)
        history['player1_noise'].append(agent1.noise_scale)
        history['player2_noise'].append(agent2.noise_scale)

        # 3. Generate and save plot
        self.save_training_plot()

    def save_training_plot(self):
        """Generates and saves a matplotlib plot of the training progress."""
        if not MATPLOTLIB_AVAILABLE:
            if self.game_number % 20 == 0: # Avoid spamming the console
                 print("Matplotlib not found. Skipping plot generation. Install with: pip install matplotlib")
            return

        history = self.plot_history
        if not history['games']: return

        try:
            import pandas as pd
            PANDAS_AVAILABLE = True
        except ImportError:
            PANDAS_AVAILABLE = False

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f'Training Progress After Game {self.game_number + 1}', fontsize=16)

        def smooth(y, box_pts):
            if len(y) < box_pts or box_pts < 1: return y
            box = np.ones(box_pts)/box_pts
            # Use slicing to handle convolution boundary issues
            return np.convolve(y, box, mode='valid')

        num_games = len(history['games'])
        smoothing_window = max(1, min(num_games // 5, 20))

        # --- Plot 1: Average Rewards ---
        games_axis = np.array(history['games'])
        p1_rewards = np.array(history['player1_avg_reward'])
        p2_rewards = np.array(history['player2_avg_reward'])
        
        axs[0].plot(games_axis, p1_rewards, 'o-', label='P1 Avg Reward (Raw)', color='C1', alpha=0.3, markersize=3)
        axs[0].plot(games_axis, p2_rewards, 'o-', label='P2 Avg Reward (Raw)', color='C0', alpha=0.3, markersize=3)

        if num_games >= smoothing_window:
            smoothed_games = games_axis[smoothing_window-1:]
            axs[0].plot(smoothed_games, smooth(p1_rewards, smoothing_window), '-', label='P1 Avg Reward (Smoothed)', color='C1', linewidth=2)
            axs[0].plot(smoothed_games, smooth(p2_rewards, smoothing_window), '-', label='P2 Avg Reward (Smoothed)', color='C0', linewidth=2)

        axs[0].set_ylabel('Average Reward (over last 500 steps)')
        axs[0].legend(loc='upper left'); axs[0].set_title('Agent Performance')

        # --- Plot 2: Training Loss ---
        p1_loss = history['player1_loss']; p2_loss = history['player2_loss']
        if PANDAS_AVAILABLE:
            p1_loss = pd.Series(p1_loss).interpolate(method='linear', limit_direction='forward').fillna(method='bfill')
            p2_loss = pd.Series(p2_loss).interpolate(method='linear', limit_direction='forward').fillna(method='bfill')
        
        axs[1].plot(games_axis, p1_loss, '-', label='Player 1 Loss', color='C1', alpha=0.8, linewidth=1.5)
        axs[1].plot(games_axis, p2_loss, '-', label='Player 2 Loss', color='C0', alpha=0.8, linewidth=1.5)
        axs[1].set_ylabel('SmoothL1Loss'); axs[1].set_yscale('log'); axs[1].legend(loc='upper left')
        axs[1].set_title('DQN Training Loss (Log Scale)')

        # --- Plot 3: Noise Scale ---
        axs[2].plot(games_axis, history['player1_noise'], '-', label='Noise Scale', color='C4')
        axs[2].set_xlabel(f'Game Number (Total Frames: {self.TOTAL_FRAMES})'); axs[2].set_ylabel('Noise Scale')
        axs[2].legend(loc='upper right'); axs[2].set_title('Parameter Space Noise Decay')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        save_dir = "screenshots"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, "training_progress.png")); plt.close(fig)

    def run_headless_simulation(self):
        """Runs the main simulation loop for a specified number of games."""
        self.start_simulation()
        while True:
            app.step()
            a_different_time.sleep(0.01)


    def end_simulation(self):
        """Cleans up and exits the simulation."""
        print(f"--- Simulation of {GAME_CONFIG['NUM_GAMES_TO_RUN']} games complete. Saving final models. ---")
        self.agent_manager.save_final_models()
        sys.exit()

    
    def handle_input(self, key):
        if key == 'escape':
            print("--- Simulation ended by user. Saving models... ---")
            self.agent_manager.save_final_models()
            sys.exit()

# --- URSINA HOOKS & MAIN EXECUTION ---
game_manager = None

def update():
    """Global update function for Ursina."""
    if game_manager:
        game_manager.update()

def input(key):
    """Global input handler for Ursina."""
    if game_manager:
        game_manager.handle_input(key)

if __name__ == '__main__':
    game_manager = GameManager()
    if GAME_CONFIG['SHOULD_RENDER']:
        app.run()
    else:
        game_manager.run_headless_simulation()