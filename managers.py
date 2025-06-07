from ursina import Vec3, color
from config import GAME_CONFIG, DQN_CONFIG, ACTIONS, PHYSICS_CONFIG
from ai import DQNAgent, device
from game import Player, Ball
import torch


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

    def reset_episode(self):
        """Resets all entities to their starting positions for a new episode."""
        self.ball.reset(position=Vec3(0, 0, 0))
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

    def _get_state_for_agent(self, player, opponent, ball, own_goal, opp_goal):
        """Constructs the state vector for a given agent's perspective."""
        norm_w = GAME_CONFIG['FIELD_WIDTH'] / 2
        norm_l = GAME_CONFIG['FIELD_LENGTH'] / 2
        p_pos = player.position

        vec_to_ball = (ball.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_opp_goal = (opp_goal.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_own_goal = (own_goal.position - p_pos) / Vec3(norm_w, 1, norm_l)
        vec_to_opponent = (opponent.position - p_pos) / Vec3(norm_w, 1, norm_l)

        ball_vel = ball.velocity / PHYSICS_CONFIG['KICK_STRENGTH']
        p_fwd = player.forward

        max_speed = PHYSICS_CONFIG.get('PLAYER_MAX_SPEED', 15)
        p_vel = player.velocity / max_speed if max_speed > 0 else Vec3(0, 0, 0)

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
            agent.memory.push(states[i], actions[i], next_states[i], reward_tensor)

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