from ursina import *
import sys
import time as a_different_time # Use a different alias to avoid conflict with ursina's time module
import torch

from config import GAME_CONFIG, DQN_CONFIG, ACTIONS
from ai import DQNAgent, device
from game import (
    setup_field, Player, Ball,
    move_player, handle_player_collisions,
    apply_kick_force, calculate_rewards
)
from ursina import Vec3
from config import PHYSICS_CONFIG


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


class SoccerSimulation:
    """
    Manages the entire soccer simulation, including game state, agents,
    and the main game loop.
    """
    def __init__(self):
        # Game State
        self.score = {'player1': 0, 'player2': 0}
        self.game_over = False
        self.time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
        self.game_number = 0
        self.episode_frame_count = 0
        self.TOTAL_FRAMES = 0
        self.last_dists_to_ball = {'player1': None, 'player2': None}

        # Game Entities (placeholders, will be populated in setup_episode)
        self.players = [None, None]
        self.opponents = [None, None]

        # UI Elements
        self.score_display = Text(origin=(0,0), y=0.4, scale=1.5, background=True)
        self.timer_display = Text("00:00", origin=(0,0), y=0.35, scale=1.2, background=True)
        
        # Environment & Entities
        self.player1_goal, self.player2_goal = setup_field()
        self.agents = []
        self.ball = None


    def setup_episode(self):
        """
        Resets the environment for a new episode (e.g., after a goal).
        It preserves the agents' learning progress while resetting their positions.
        """
        # Store learning progress from old agents before reset
        learning_progress = {}
        for agent in self.agents:
            learning_progress[agent.team_name] = {
                'policy_net_state': agent.policy_net.state_dict(),
                'target_net_state': agent.target_net.state_dict(),
                'memory': agent.memory,
                'steps_done': agent.steps_done
            }
            if agent.player is not None and agent.player in scene.entities:
                destroy(agent.player)

        if self.ball:
            destroy(self.ball)

        self.ball = Ball(position=(0,0,0))
        
        self.players = [
            Player(position=(-15, 0, 0), color=color.orange, rotation_y=90),
            Player(position=(15, 0, 0), color=color.blue, rotation_y=-90)
        ]
        self.opponents = [self.players[1], self.players[0]]
    
        # Re-create agents and load their progress
        new_agents = []
        goal_assignments = [(self.player1_goal, self.player2_goal), (self.player2_goal, self.player1_goal)]
        team_names = ['player1', 'player2']
        
        # Determine state size from a sample state
        p1_own_goal, p1_opp_goal = goal_assignments[0]
        sample_state = self._get_state_for_agent(self.players[0], self.opponents[0], self.ball, p1_own_goal, p1_opp_goal)
        state_size = len(sample_state)
        action_size = len(ACTIONS)

        for i in range(2):
            team_name = team_names[i]
            agent = DQNAgent(team_name, DQN_CONFIG, state_size, action_size)
            
            # Link agent to its player entity and its opponent's goal
            agent.player = self.players[i]
            agent.opp_goal = goal_assignments[i][1]

            # Restore learning progress
            if team_name in learning_progress:
                progress = learning_progress[team_name]
                agent.policy_net.load_state_dict(progress['policy_net_state'])
                agent.target_net.load_state_dict(progress['target_net_state'])
                agent.memory = progress['memory']
                agent.steps_done = progress['steps_done']
            else: # Load from file on first ever run
                agent.load_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{team_name}.pth")

            new_agents.append(agent)
        
        self.agents = new_agents
        self.last_dists_to_ball = {name: None for name in team_names}


    def _get_state_for_agent(self, player, opponent, ball, own_goal, opp_goal):
        """Constructs the state vector for a given agent's perspective."""
        norm_w = GAME_CONFIG['FIELD_WIDTH'] / 2
        norm_l = GAME_CONFIG['FIELD_LENGTH'] / 2
        p_pos = player.position

        # Relative vectors, normalized
        vec_to_ball = (ball.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_opp_goal = (opp_goal.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_own_goal = (own_goal.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_opponent = (opponent.position - p_pos) / Vec3(norm_w, 1, norm_l)

        ball_vel = ball.velocity / PHYSICS_CONFIG['KICK_STRENGTH']
        p_fwd = player.forward

        # Player velocity, normalized
        max_speed = PHYSICS_CONFIG.get('PLAYER_MAX_SPEED', 15)
        p_vel = player.velocity / max_speed if max_speed > 0 else Vec3(0,0,0)

        # Dot products to represent angles
        angle_to_ball = p_fwd.dot(vec_to_ball.normalized())
        angle_to_opp_goal = p_fwd.dot(vec_to_opp_goal.normalized())

        state = [
            vec_to_ball.x, vec_to_ball.z,
            ball_vel.x, ball_vel.z,
            vec_to_opp_goal.x, vec_to_opp_goal.z,
            vec_to_own_goal.x, vec_to_own_goal.z,
            vec_to_opponent.x, vec_to_opponent.z,
            p_fwd.x, p_fwd.z,
            p_vel.x, p_vel.z,
            angle_to_ball,
            angle_to_opp_goal
        ]
        assert len(state) == DQN_CONFIG['STATE_SIZE'], "State vector length does not match config"
        return state


    def start_new_game(self):
        """
        Initializes a new game, resetting scores and timers.
        """
        print(f"--- Starting Game {self.game_number + 1}/{GAME_CONFIG['NUM_GAMES_TO_RUN']} ---")
        self.score = {'player1': 0, 'player2': 0}
        self.time_left = GAME_CONFIG['GAME_TIMER_SECONDS']
        self.game_over = False
        self.setup_episode()
        self.episode_frame_count = 0
        self.update_score_ui()
        if GAME_CONFIG['SHOULD_RENDER']:
            self.update_timer_ui()
            # Workaround for a rendering issue on macOS where the window may not
            # update until user interaction. This forces a redraw.
            if sys.platform == 'darwin':
                window.size += (0, 1)
                window.size -= (0, 1)


    # --- UI Methods ---
    def update_score_ui(self):
        self.score_display.text = f"<orange>{self.score['player1']}<default> - <azure>{self.score['player2']}"

    def update_timer_ui(self):
        mins, secs = divmod(self.time_left, 60)
        self.timer_display.text = f"Time: {int(mins):02}:{int(secs):02}"


    # --- Core Game Loop ---
    def update(self):
        """
        This is the main update loop, called by Ursina on every frame.
        """
        if self.game_over: return
        dt = time.dt
        if dt == 0: return

        self._update_timer(dt)
        if self.game_over: return # Game might have ended due to time out

        states, actions = self._handle_player_movement_and_collisions()
        
        # Handle ball physics and calculate all rewards in one place
        prev_ball_dists = {a.team_name: distance_xz(self.ball.position, a.opp_goal.position) for a in self.agents}
        hit_info = self.ball.intersects()
        
        if hit_info.hit:
            apply_kick_force(hit_info, self.ball, self.agents)

        rewards, done, scoring_team = calculate_rewards(
            self.agents, self.ball, self.player1_goal, self.player2_goal, 
            hit_info, prev_ball_dists, self.last_dists_to_ball
        )

        self._update_learning(states, actions, rewards, done)

        if done:
            if scoring_team:
                self.score[scoring_team] += 1
            if GAME_CONFIG['SHOULD_RENDER']: self.update_score_ui()
            self.setup_episode()
            self.episode_frame_count = 0


    def _update_timer(self, dt):
        """Handles the game timer."""
        self.time_left -= dt
        if self.time_left <= 0:
            self.time_left = 0
            self.game_over = True
        if GAME_CONFIG['SHOULD_RENDER']:
            self.update_timer_ui()

    def _handle_player_movement_and_collisions(self):
        """Gets actions from agents, applies them, and handles player-player collisions."""
        states = []
        goal_assignments = [(self.player1_goal, self.player2_goal), (self.player2_goal, self.player1_goal)]
        for i in range(len(self.agents)):
            own_goal, opp_goal = goal_assignments[i]
            state_list = self._get_state_for_agent(
                self.players[i], self.opponents[i], self.ball, own_goal, opp_goal
            )
            states.append(torch.tensor(state_list, dtype=torch.float32, device=device).unsqueeze(0))

        actions = [agent.select_action(state) for agent, state in zip(self.agents, states)]
        for i, action in enumerate(actions):
            move_player(self.players[i], action)

        handle_player_collisions(self.players[0], self.players[1])
        return states, actions

    def _update_learning(self, states, actions, rewards, done):
        """Handles the learning step for each agent and model checkpointing."""
        next_states = []
        if not done:
            goal_assignments = [(self.player1_goal, self.player2_goal), (self.player2_goal, self.player1_goal)]
            for i in range(len(self.agents)):
                own_goal, opp_goal = goal_assignments[i]
                state_list = self._get_state_for_agent(
                    self.players[i], self.opponents[i], self.ball, own_goal, opp_goal
                )
                next_states.append(torch.tensor(state_list, dtype=torch.float32, device=device).unsqueeze(0))
        else:
            next_states = [None] * len(self.agents)

        for i, agent in enumerate(self.agents):
            reward_tensor = torch.tensor([rewards[agent.team_name]], device=device)
            agent.memory.push(states[i], actions[i], next_states[i], reward_tensor)

        self.TOTAL_FRAMES += 1
        self.episode_frame_count += 1

        if self.TOTAL_FRAMES % DQN_CONFIG['UPDATE_EVERY'] == 0:
            for agent in self.agents:
                agent.optimize_model()

        if self.TOTAL_FRAMES % DQN_CONFIG['TARGET_UPDATE_EVERY'] == 0:
            for agent in self.agents:
                agent.update_target_net()

        if self.TOTAL_FRAMES > 0 and self.TOTAL_FRAMES % DQN_CONFIG['CHECKPOINT_EVERY'] == 0:
            for agent in self.agents:
                agent.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{agent.team_name}_{self.TOTAL_FRAMES}.pth")
    
    # --- Simulation Runner ---
    def run_simulation(self, num_games_to_play):
        """
        Runs the main simulation loop for a specified number of games.
        """
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
        self.save_final_models()
        sys.exit()

    def save_final_models(self):
        for agent in self.agents:
            agent.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{agent.team_name}_final.pth")
    
    def handle_input(self, key):
        if key == 'escape':
            print("--- Simulation ended by user. Saving models... ---")
            self.save_final_models()
            sys.exit()

# --- URSINA HOOKS & MAIN EXECUTION ---
simulation = None

def update():
    """Global update function for Ursina."""
    if simulation:
        simulation.update()

def input(key):
    """Global input handler for Ursina."""
    if simulation:
        simulation.handle_input(key)

if __name__ == '__main__':
    simulation = SoccerSimulation()
    simulation.run_simulation(GAME_CONFIG['NUM_GAMES_TO_RUN'])