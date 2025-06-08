from ursina import *
import sys
import time as a_different_time # Use a different alias to avoid conflict with ursina's time module

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