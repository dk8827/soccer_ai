from ursina import Vec3, color, lerp
from config import GAME_CONFIG, DQN_CONFIG, ACTIONS, PHYSICS_CONFIG, CURRICULUM_CONFIG
from ai import DQNAgent, device
from game import Player, Ball
import torch
import math
import random


def rotate_vector(vec, angle_rad):
    """Rotates a 2D vector (represented as a Vec3 on the XZ plane) by a given angle."""
    x = vec.x * math.cos(angle_rad) - vec.z * math.sin(angle_rad)
    z = vec.x * math.sin(angle_rad) + vec.z * math.cos(angle_rad)
    return Vec3(x, vec.y, z)


class EntityManager:
    """Manages the lifecycle of game entities (players, ball)."""
    def __init__(self, ground, player1_goal, player2_goal):
        self.ground = ground
        self.player1_goal = player1_goal
        self.player2_goal = player2_goal

        self.ball = Ball(position=(0, 0, 0), ground=self.ground)
        self.players = [
            Player(position=(-15, 0, 0), color=color.orange, rotation_y=90, ground=self.ground),
            Player(position=(15, 0, 0), color=color.blue, rotation_y=-90, ground=self.ground)
        ]
        self.opponents = [self.players[1], self.players[0]]

    def reset_episode(self, total_frames=0):
        """Resets all entities to their starting positions for a new episode."""
        if CURRICULUM_CONFIG.get('ENABLED', False):
            max_z_end = GAME_CONFIG['FIELD_LENGTH'] / CURRICULUM_CONFIG['BALL_Z_RANGE_END_FACTOR']
            
            schedule_frames = CURRICULUM_CONFIG['SCHEDULE_FRAMES']
            progress = 1.0
            if schedule_frames > 0:
                progress = min(1.0, total_frames / schedule_frames)

            current_max_z = lerp(CURRICULUM_CONFIG['BALL_Z_RANGE_START'], max_z_end, progress)
        else:
            current_max_z = GAME_CONFIG['FIELD_LENGTH'] / 2.5 # Give some margin from the walls

        random_z = random.uniform(-current_max_z, current_max_z)
        self.ball.reset(position=Vec3(0, 0, random_z))
        self.players[0].reset(position=Vec3(-15, 0, 0), rotation_y=90)
        self.players[1].reset(position=Vec3(15, 0, 0), rotation_y=-90)

class AgentManager:
    """Manages the AI agents, including their creation, actions, and learning."""
    def __init__(self, entity_manager, player1_goal, player2_goal):
        self.entity_manager = entity_manager
        self.agents = []
        self.last_dists_to_ball = {}

        goal_assignments = [(player1_goal, player2_goal), (player2_goal, player1_goal)]
        team_names = ['player1', 'player2']
        
        sample_state = self._get_state_for_agent(
            self.entity_manager.players[0], self.entity_manager.opponents[0],
            self.entity_manager.ball, goal_assignments[0][0], goal_assignments[0][1]
        )
        state_size = len(sample_state)
        action_size = len(ACTIONS)

        for i in range(2):
            team_name = team_names[i]
            agent = DQNAgent(team_name, DQN_CONFIG, state_size, action_size)
            agent.player = self.entity_manager.players[i]
            agent.opp_goal = goal_assignments[i][1]
            agent.load_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{team_name}.pth")
            self.agents.append(agent)

        self.reset_episode()

    def reset_episode(self):
        """Resets the state for the agents for a new episode."""
        self.last_dists_to_ball = {agent.team_name: None for agent in self.agents}
        # Apply new noise for the upcoming episode
        for agent in self.agents:
            agent.apply_noise()

    def _get_state_for_agent(self, player, opponent, ball, own_goal, opp_goal):
        """
        Constructs a player-centric state vector for a given agent.
        All vectors are rotated relative to the player's orientation.
        """
        # Normalization factors
        norm_w = GAME_CONFIG['FIELD_WIDTH'] / 2
        norm_l = GAME_CONFIG['FIELD_LENGTH'] / 2
        max_speed = PHYSICS_CONFIG.get('PLAYER_MAX_SPEED', 15)
        
        p_pos = player.position
        # Angle for rotating world vectors into player's local coordinate system.
        # Negative angle because we are rotating the world, not the player.
        player_angle_rad = -math.radians(player.rotation_y)

        # 1. Calculate world-frame vectors from player to objects and normalize
        vec_to_ball_world = (ball.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_opp_goal_world = (opp_goal.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_own_goal_world = (own_goal.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_opponent_world = (opponent.position - p_pos) / Vec3(norm_w, 1, norm_l)

        # 2. Rotate these vectors into player's local reference frame
        vec_to_ball = rotate_vector(vec_to_ball_world, player_angle_rad)
        vec_to_opp_goal = rotate_vector(vec_to_opp_goal_world, player_angle_rad)
        vec_to_own_goal = rotate_vector(vec_to_own_goal_world, player_angle_rad)
        vec_to_opponent = rotate_vector(vec_to_opponent_world, player_angle_rad)

        # 3. Handle velocities (which are direction vectors)
        ball_vel_world = ball.velocity / PHYSICS_CONFIG['KICK_STRENGTH']
        p_vel_world = player.velocity / max_speed if max_speed > 0 else Vec3(0, 0, 0)
        
        ball_vel = rotate_vector(ball_vel_world, player_angle_rad)
        p_vel = rotate_vector(p_vel_world, player_angle_rad)

        # In the player's reference frame, their own forward vector is always (0, 0, 1),
        # and angles to objects are implicitly captured by the rotated vectors.
        # Thus, p_fwd, angle_to_ball, and angle_to_opp_goal are no longer needed.
        
        state = [
            vec_to_ball.x, vec_to_ball.z,
            ball_vel.x, ball_vel.z,
            vec_to_opp_goal.x, vec_to_opp_goal.z,
            vec_to_own_goal.x, vec_to_own_goal.z,
            vec_to_opponent.x, vec_to_opponent.z,
            p_vel.x, p_vel.z,
        ]
        return state

    def get_actions_and_states(self):
        """Gets actions from agents for the current state."""
        states = []
        goal_assignments = [(self.agents[0].opp_goal, self.agents[1].opp_goal), (self.agents[1].opp_goal, self.agents[0].opp_goal)]
        
        for i, agent in enumerate(self.agents):
            own_goal, opp_goal = (self.entity_manager.player1_goal, self.entity_manager.player2_goal) if agent.team_name == 'player1' else (self.entity_manager.player2_goal, self.entity_manager.player1_goal)
            
            state_list = self._get_state_for_agent(
                agent.player, self.entity_manager.opponents[i], self.entity_manager.ball, own_goal, opp_goal
            )
            states.append(torch.tensor(state_list, dtype=torch.float32, device=device).unsqueeze(0))

        actions = [agent.select_action(state) for agent, state in zip(self.agents, states)]
        return states, actions

    def update_learning(self, states, actions, rewards, done, total_frames):
        """Handles the learning step for each agent and model checkpointing."""
        next_states = []
        if not done:
            goal_assignments = [(self.agents[0].opp_goal, self.agents[1].opp_goal), (self.agents[1].opp_goal, self.agents[0].opp_goal)]
            for i, agent in enumerate(self.agents):
                own_goal, opp_goal = (self.entity_manager.player1_goal, self.entity_manager.player2_goal) if agent.team_name == 'player1' else (self.entity_manager.player2_goal, self.entity_manager.player1_goal)
                state_list = self._get_state_for_agent(
                    agent.player, self.entity_manager.opponents[i], self.entity_manager.ball, own_goal, opp_goal
                )
                next_states.append(torch.tensor(state_list, dtype=torch.float32, device=device).unsqueeze(0))
        else:
            next_states = [None] * len(self.agents)

        for i, agent in enumerate(self.agents):
            reward_tensor = torch.tensor([rewards[agent.team_name]], device=device)
            current_reward = rewards[agent.team_name]
            agent.memory.push(states[i], actions[i], next_states[i], reward_tensor)
            agent.max_reward = max(agent.max_reward, current_reward)
            agent.reward_history.append(current_reward)

        if total_frames % DQN_CONFIG['UPDATE_EVERY'] == 0:
            for agent in self.agents:
                agent.optimize_model()

        if total_frames % DQN_CONFIG['TARGET_UPDATE_EVERY'] == 0:
            for agent in self.agents:
                agent.update_target_net()

        if total_frames > 0 and total_frames % DQN_CONFIG['CHECKPOINT_EVERY'] == 0:
            for agent in self.agents:
                agent.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{agent.team_name}_{total_frames}.pth")

    def save_final_models(self):
        """Saves the final trained models for all agents."""
        for agent in self.agents:
            agent.save_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{agent.team_name}_final.pth") 