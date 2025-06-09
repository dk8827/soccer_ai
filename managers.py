from ursina import Vec3, color, lerp
from config import GAME_CONFIG, DQN_CONFIG, ACTIONS, MACRO_ACTIONS, PHYSICS_CONFIG, CURRICULUM_CONFIG
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
        self.ball_start_player_index = 0

    def reset_episode(self, total_frames=0):
        """
        Resets all entities to their starting positions for a new episode.
        Implements curriculum learning to transition from a player starting with the ball
        to a standard, neutral kickoff.
        """
        if not CURRICULUM_CONFIG.get('ENABLED', False):
            # Fallback to a simple random reset if curriculum is disabled
            current_max_z = GAME_CONFIG['FIELD_LENGTH'] / 2.5
            random_z = random.uniform(-current_max_z, current_max_z)
            self.ball.reset(position=Vec3(0, 0, random_z))
            self.players[0].reset(position=Vec3(-15, 0, 0), rotation_y=90)
            self.players[1].reset(position=Vec3(15, 0, 0), rotation_y=-90)
            return

        # --- CURRICULUM LEARNING LOGIC ---
        schedule_frames = CURRICULUM_CONFIG.get('SCHEDULE_FRAMES', 500000)
        progress = min(1.0, total_frames / schedule_frames) if schedule_frames > 0 else 1.0

        # --- Stage 0: Player starts with the ball (at progress = 0) ---
        player_with_ball_idx = self.ball_start_player_index
        p_without_ball_idx = 1 - player_with_ball_idx

        field_w_half = GAME_CONFIG['FIELD_WIDTH'] / 2
        field_l_half = GAME_CONFIG['FIELD_LENGTH'] / 2

        # Position for player starting with the ball (in their own half)
        p_with_ball_x = random.uniform(-field_w_half * 0.8, -2) if player_with_ball_idx == 0 else random.uniform(2, field_w_half * 0.8)
        p_with_ball_z = random.uniform(-field_l_half * 0.8, field_l_half * 0.8)
        p_with_ball_pos_start = Vec3(p_with_ball_x, 0, p_with_ball_z)

        # Ball is placed between player and opponent goal
        ball_offset = Vec3(2.0, 0, 0) if player_with_ball_idx == 0 else Vec3(-2.0, 0, 0)
        ball_pos_start = p_with_ball_pos_start + ball_offset

        # Position for player without the ball (in their own half)
        p_without_ball_x = random.uniform(2, field_w_half * 0.8) if p_without_ball_idx == 1 else random.uniform(-field_w_half * 0.8, -2)
        p_without_ball_z = random.uniform(-field_l_half * 0.8, field_l_half * 0.8)
        p_without_ball_pos_start = Vec3(p_without_ball_x, 0, p_without_ball_z)
        
        # Assign starting positions based on who has the ball
        p1_pos_start = p_with_ball_pos_start if player_with_ball_idx == 0 else p_without_ball_pos_start
        p2_pos_start = p_without_ball_pos_start if player_with_ball_idx == 0 else p_with_ball_pos_start

        # --- Stage 1: Standard kickoff (at progress = 1) ---
        max_z_end = GAME_CONFIG['FIELD_LENGTH'] / CURRICULUM_CONFIG.get('BALL_Z_RANGE_END_FACTOR', 2.5)
        random_z_end = random.uniform(-max_z_end, max_z_end)

        ball_pos_end = Vec3(0, 0, random_z_end)
        p1_pos_end = Vec3(-15, 0, 0)
        p2_pos_end = Vec3(15, 0, 0)

        # --- Interpolate between stages based on progress ---
        final_ball_pos = lerp(ball_pos_start, ball_pos_end, progress)
        final_p1_pos = lerp(p1_pos_start, p1_pos_end, progress)
        final_p2_pos = lerp(p2_pos_start, p2_pos_end, progress)

        self.ball.reset(position=final_ball_pos)
        self.players[0].reset(position=final_p1_pos, rotation_y=90)
        self.players[1].reset(position=final_p2_pos, rotation_y=-90)

        # Alternate which player starts with the ball for the next episode
        self.ball_start_player_index = 1 - self.ball_start_player_index

class AgentManager:
    """Manages the AI agents, including their creation, actions, and learning."""
    def __init__(self, entity_manager, player1_goal, player2_goal):
        self.entity_manager = entity_manager
        self.agents = []
        self.agent_trackers = []

        goal_assignments = [(player1_goal, player2_goal), (player2_goal, player1_goal)]
        team_names = ['player1', 'player2']
        
        # Determine state and action sizes from a sample
        sample_state = self._get_state_for_agent(
            self.entity_manager.players[0], self.entity_manager.opponents[0],
            self.entity_manager.ball, goal_assignments[0][0], goal_assignments[0][1]
        )
        state_size = len(sample_state)
        action_size = len(MACRO_ACTIONS) # Use the number of MACRO actions

        for i in range(2):
            team_name = team_names[i]
            agent = DQNAgent(team_name, DQN_CONFIG, state_size, action_size)
            agent.player = self.entity_manager.players[i]
            agent.opp_goal = goal_assignments[i][1]
            agent.own_goal = goal_assignments[i][0]
            agent.load_model(GAME_CONFIG['SAVE_DIR'], f"dqn_soccer_{team_name}.pth")
            self.agents.append(agent)
            
            # Initialize a tracker for each agent to handle macro actions
            self.agent_trackers.append({
                'current_macro_info': None, # (primitive_action_id, duration)
                'frames_left': 0,
                'reward_buffer': 0.0,
                'last_state': None,
                'last_macro_action_id': None,
                'last_macro_duration': 0,
                'frames_executed_in_macro': 0,
            })

        self.reset_episode()

    def reset_episode(self):
        """Resets the state for the agents for a new episode."""
        for agent in self.agents:
            agent.position_history.clear()
        
        for tracker in self.agent_trackers:
            tracker['current_macro_info'] = None
            tracker['frames_left'] = 0
            tracker['reward_buffer'] = 0.0
            tracker['last_state'] = None
            tracker['last_macro_action_id'] = None
            tracker['last_macro_duration'] = 0
            tracker['frames_executed_in_macro'] = 0

    def _get_wall_distance(self, player):
        """
        Calculates the distance from the player to the nearest wall in the direction they're facing.
        Returns a normalized distance (0 to 1) where 1 is at the wall.
        """
        # Get field dimensions
        field_width = GAME_CONFIG['FIELD_WIDTH']
        field_length = GAME_CONFIG['FIELD_LENGTH']
        
        # Get player's position and forward direction
        pos = player.position
        forward = player.forward.normalized() # Ensure the forward vector is normalized
        
        distances = []
        # Calculate distance to each wall plane along the forward vector
        if abs(forward.x) > 1e-6:
            # Time/distance to hit vertical walls (x-planes)
            t_to_pos_x_wall = (field_width/2 - pos.x) / forward.x
            t_to_neg_x_wall = (-field_width/2 - pos.x) / forward.x
            if t_to_pos_x_wall >= 0: distances.append(t_to_pos_x_wall)
            if t_to_neg_x_wall >= 0: distances.append(t_to_neg_x_wall)

        if abs(forward.z) > 1e-6:
            # Time/distance to hit horizontal walls (z-planes)
            t_to_pos_z_wall = (field_length/2 - pos.z) / forward.z
            t_to_neg_z_wall = (-field_length/2 - pos.z) / forward.z
            if t_to_pos_z_wall >= 0: distances.append(t_to_pos_z_wall)
            if t_to_neg_z_wall >= 0: distances.append(t_to_neg_z_wall)

        # Get the minimum positive distance (the closest intersection in front of the player)
        wall_dist = min(distances) if distances else float('inf')

        # Normalize the distance (0 near, 1 far)
        # We use a large distance to represent "no wall in sight"
        max_dist = max(field_width, field_length)
        normalized_dist = 1.0 - clamp(wall_dist / max_dist, 0, 1) # Invert so 1 is at wall, 0 is far

        return normalized_dist

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
        opp_vel_world = opponent.velocity / max_speed if max_speed > 0 else Vec3(0, 0, 0)
        
        ball_vel = rotate_vector(ball_vel_world, player_angle_rad)
        p_vel = rotate_vector(p_vel_world, player_angle_rad)
        opp_vel = rotate_vector(opp_vel_world, player_angle_rad)

        # 4. Calculate wall distance
        wall_dist = self._get_wall_distance(player)
        
        state = [
            vec_to_ball.x, vec_to_ball.z,
            ball_vel.x, ball_vel.z,
            vec_to_opp_goal.x, vec_to_opp_goal.z,
            vec_to_own_goal.x, vec_to_own_goal.z,
            vec_to_opponent.x, vec_to_opponent.z,
            p_vel.x, p_vel.z,
            opp_vel.x, opp_vel.z,
            wall_dist,  # Add wall distance to state
        ]
        return state

    def get_primitive_actions_and_states(self):
        """
        Decides on an action for each agent. If a macro is in progress, it continues it.
        If not, it selects a new macro action.
        This function is also responsible for pushing completed macro transitions to memory.
        """
        primitive_actions = []
        states_for_main = [] # The 'main' loop needs states for the update_learning call, even if we don't use them there.
        
        for i, agent in enumerate(self.agents):
            tracker = self.agent_trackers[i]

            # --- Decision Making ---
            if tracker['frames_left'] <= 0:
                # --- Learning Step for the previous action ---
                if tracker['last_state'] is not None:
                    next_state = self._get_state_for_agent(
                        agent.player, self.entity_manager.opponents[i], self.entity_manager.ball,
                        agent.own_goal, agent.opp_goal
                    )
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    reward_tensor = torch.tensor([tracker['reward_buffer']], device=device)
                    duration_tensor = torch.tensor([tracker['last_macro_duration']], dtype=torch.float, device=device)
                    
                    agent.memory.push(tracker['last_state'], tracker['last_macro_action_id'], next_state_tensor, reward_tensor, duration_tensor)
                    
                # --- Select New Macro Action ---
                agent.apply_noise()
                current_state = self._get_state_for_agent(
                    agent.player, self.entity_manager.opponents[i], self.entity_manager.ball,
                    agent.own_goal, agent.opp_goal
                )
                current_state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
                
                macro_action_tensor = agent.select_action(current_state_tensor)
                macro_action_id = macro_action_tensor.item()
                
                # Update tracker for the new macro
                tracker['last_state'] = current_state_tensor
                tracker['last_macro_action_id'] = macro_action_tensor
                tracker['current_macro_info'] = MACRO_ACTIONS[macro_action_id]
                tracker['last_macro_duration'] = tracker['current_macro_info'][1]
                tracker['frames_left'] = tracker['current_macro_info'][1]
                tracker['reward_buffer'] = 0.0 # Reset reward buffer for the new macro
                tracker['frames_executed_in_macro'] = 0
            
            # --- Execute Current Action ---
            tracker['frames_left'] -= 1
            primitive_action_id = tracker['current_macro_info'][0]
            primitive_actions.append(torch.tensor([[primitive_action_id]], device=device))
            states_for_main.append(tracker['last_state']) # Pass the state that led to the current macro

        return states_for_main, primitive_actions

    def step_learning_update(self, rewards, done, total_frames):
        """
        Accumulates rewards, and if the episode is done, it forces the final
        transition to be stored in memory. Also triggers the optimization step.
        """
        for i, agent in enumerate(self.agents):
            agent.steps_done = total_frames
            # Accumulate reward, discounted by gamma for each step within the macro
            reward_val = rewards[agent.team_name]
            gamma = DQN_CONFIG['GAMMA']
            discount = gamma ** self.agent_trackers[i]['frames_executed_in_macro']
            self.agent_trackers[i]['reward_buffer'] += reward_val * discount
            self.agent_trackers[i]['frames_executed_in_macro'] += 1

            agent.reward_history.append(reward_val) # Keep original reward history for stats
            agent.max_reward = max(agent.max_reward, reward_val)

        if done:
            # Episode ended. Store the final transition.
            for i, agent in enumerate(self.agents):
                tracker = self.agent_trackers[i]
                if tracker['last_state'] is not None:
                    reward_tensor = torch.tensor([tracker['reward_buffer']], device=device)
                    # For the final transition, the duration is the number of frames actually executed
                    duration_executed = tracker['frames_executed_in_macro']
                    duration_tensor = torch.tensor([duration_executed], dtype=torch.float, device=device)
                    # The next_state is None because the episode terminated
                    agent.memory.push(tracker['last_state'], tracker['last_macro_action_id'], None, reward_tensor, duration_tensor)
                    # By resetting the last_state, we prevent the completed
                    # macro transition from being added in the next call to get_primitive_actions_and_states.
                    tracker['last_state'] = None

        # Optimization and target network updates still happen on a per-frame basis
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