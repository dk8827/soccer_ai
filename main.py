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
        self.game_over = False
        self.time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
        self.game_number = 0
        self.episode_frame_count = 0
        self.TOTAL_FRAMES = 0

        # UI & Environment
        self.ui_manager = UIManager()
        player1_goal, player2_goal, ground = setup_field()
        
        # Managers
        self.entity_manager = EntityManager(ground, player1_goal, player2_goal)
        self.agent_manager = AgentManager(self.entity_manager, player1_goal, player2_goal)


    def start_new_game(self):
        """Initializes a new game, resetting scores and timers."""
        print(f"--- Starting Game {self.game_number + 1}/{GAME_CONFIG['NUM_GAMES_TO_RUN']} ---")
        self.score = {'player1': 0, 'player2': 0}
        self.time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
        self.game_over = False
        self.episode_frame_count = 0
        self.ui_manager.update_score(self.score)
        self.ui_manager.update_timer(self.time_left)
        
        self.entity_manager.reset_episode()
        self.agent_manager.reset_episode()

        if GAME_CONFIG['SHOULD_RENDER'] and sys.platform == 'darwin':
            window.size += (0, 1); window.size -= (0, 1)


    def update(self):
        """Main update loop, called by Ursina every frame."""
        if self.game_over: return
        dt = time.dt
        if dt == 0: return

        self._update_timer(dt)
        if self.game_over: return

        # 1. Get AI actions and move players
        states, actions = self.agent_manager.get_actions_and_states()
        for i, action in enumerate(actions):
            move_player(self.entity_manager.players[i], action)
        handle_player_collisions(self.entity_manager.players[0], self.entity_manager.players[1])

        # 2. Handle ball physics and interactions
        ball = self.entity_manager.ball
        prev_ball_dists = {a.team_name: distance_xz(ball.position, a.opp_goal.position) for a in self.agent_manager.agents}
        hit_info = ball.intersects(self.entity_manager.players[0]) or ball.intersects(self.entity_manager.players[1])
        
        if hit_info.hit:
            apply_kick_force(hit_info, ball, self.agent_manager.agents)

        # 3. Calculate rewards
        ctx = RewardContext(
            agents=self.agent_manager.agents,
            ball=ball,
            player1_goal=self.entity_manager.player1_goal,
            player2_goal=self.entity_manager.player2_goal,
            hit_info=hit_info,
            prev_ball_dists=prev_ball_dists,
            last_dists_to_ball=self.agent_manager.last_dists_to_ball
        )
        rewards, done, scoring_team, new_dists_to_ball = calculate_rewards(ctx)
        self.agent_manager.last_dists_to_ball = new_dists_to_ball

        # 4. Update AI learning
        self.TOTAL_FRAMES += 1
        self.episode_frame_count += 1
        self.agent_manager.update_learning(states, actions, rewards, done, self.TOTAL_FRAMES)

        # 5. Check for episode end
        if done:
            if scoring_team:
                self.score[scoring_team] += 1
            self.ui_manager.update_score(self.score)
            self.entity_manager.reset_episode()
            self.agent_manager.reset_episode()
            self.episode_frame_count = 0


    def _update_timer(self, dt):
        """Handles the game timer."""
        self.time_left -= dt
        if self.time_left <= 0:
            self.time_left = 0
            self.game_over = True
        self.ui_manager.update_timer(self.time_left)


    def run_simulation(self, num_games_to_play):
        """Runs the main simulation loop for a specified number of games."""
        for i in range(num_games_to_play):
            self.game_number = i
            self.start_new_game()
            while not self.game_over:
                app.step()

            print(f"Game {self.game_number + 1} Finished. Final Score: Player1 {self.score['player1']} - Player2 {self.score['player2']}")
            print("-" * 40)
            if not GAME_CONFIG['SHOULD_RENDER']:
                 a_different_time.sleep(0.01)

        print(f"--- Simulation of {num_games_to_play} games complete. Saving final models. ---")
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
    game_manager.run_simulation(GAME_CONFIG['NUM_GAMES_TO_RUN'])