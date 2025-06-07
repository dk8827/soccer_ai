from ursina import *
import math
import random
from panda3d.core import Quat

from config import GAME_CONFIG, PHYSICS_CONFIG, DQN_CONFIG, ACTIONS

goal_width = 20
ground = None # Will be initialized in setup_field

def setup_field():
    """Creates the ground, walls, and goals for the soccer field."""
    global ground
    ground = Entity(model='quad', texture='assets/grid_texture.png', scale=(GAME_CONFIG['FIELD_WIDTH'], GAME_CONFIG['FIELD_LENGTH']), rotation_x=90, collider='box', y=-0.5)

    wall_texture = 'assets/wall_texture.png'
    wall_height = 12
    wall_segment_length = (GAME_CONFIG['FIELD_LENGTH'] - goal_width) / 2
    wall_z_offset = goal_width / 2 + wall_segment_length / 2

    # Side walls (visible)
    Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(-GAME_CONFIG['FIELD_WIDTH']/2 - 0.5, wall_height/2 - 0.5, -wall_z_offset), collider='box')
    Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(-GAME_CONFIG['FIELD_WIDTH']/2 - 0.5, wall_height/2 - 0.5, wall_z_offset), collider='box')
    Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(GAME_CONFIG['FIELD_WIDTH']/2 + 0.5, wall_height/2 - 0.5, -wall_z_offset), collider='box')
    Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(GAME_CONFIG['FIELD_WIDTH']/2 + 0.5, wall_height/2 - 0.5, wall_z_offset), collider='box')

    # Top/Bottom walls
    Entity(model='cube', texture=wall_texture, scale=(GAME_CONFIG['FIELD_WIDTH'] + 2, wall_height, 1), position=(0, wall_height/2 - 0.5, GAME_CONFIG['FIELD_LENGTH']/2 + 0.5), collider='box')
    Entity(model='cube', scale=(GAME_CONFIG['FIELD_WIDTH'] + 2, wall_height, 1), position=(0, wall_height/2 - 0.5, -GAME_CONFIG['FIELD_LENGTH']/2 - 0.5), collider='box', visible=False) # Invisible wall

    # Goal crossbars
    Entity(model='cube', texture=wall_texture, scale=(1, 1.25, goal_width), position=(-GAME_CONFIG['FIELD_WIDTH']/2 - 0.5, 10.875, 0), collider='box')
    Entity(model='cube', texture=wall_texture, scale=(1, 1.25, goal_width), position=(GAME_CONFIG['FIELD_WIDTH']/2 + 0.5, 10.875, 0), collider='box')

    player1_goal_entity = Goal(clr='player1', position=(-GAME_CONFIG['FIELD_WIDTH']/2 - 0.5, 0, 0), rotation_y=-90)
    player2_goal_entity = Goal(clr='player2', position=(GAME_CONFIG['FIELD_WIDTH']/2 + 0.5, 0, 0), rotation_y=90)

    return player1_goal_entity, player2_goal_entity

class Goal(Entity):
    def __init__(self, clr, **kwargs):
        super().__init__(**kwargs)
        w, h = 20, 10; recess_depth = 2; frame_color = color.orange if clr == 'player1' else color.blue
        Entity(parent=self, model='cube', collider='box', scale=(w, .5, .5), position=(0, h, recess_depth), color=frame_color)
        Entity(parent=self, model='cube', collider='box', scale=(.5, h, .5), position=(-w/2, h/2, recess_depth), color=frame_color)
        Entity(parent=self, model='cube', collider='box', scale=(.5, h, .5), position=(w/2, h/2, recess_depth), color=frame_color)
        Entity(parent=self, model='quad', texture='assets/net_texture.png', double_sided=True, texture_scale=(w/10, h/5), scale=(w,h), position=(0, h/2, recess_depth + 0.05))
        self.trigger = Entity(parent=self, model='cube', collider='box', scale=(w-0.5, h-0.5, 1), position=(0, h/2, recess_depth + 0.5), visible=False)

class Player(Entity):
    def __init__(self, **kwargs):
        super().__init__(model='cube', scale=1.8, collider='box', **kwargs)
        eye_dist = 0.25; eye_y = 0.2; eye_z_offset = 0.51
        Entity(parent=self, model='sphere', color=color.white, scale=0.3, position=(-eye_dist, eye_y, eye_z_offset))
        Entity(parent=self, model='sphere', color=color.black, scale=0.15, position=(-eye_dist, eye_y, eye_z_offset + 0.01))
        Entity(parent=self, model='sphere', color=color.white, scale=0.3, position=(eye_dist, eye_y, eye_z_offset))
        Entity(parent=self, model='sphere', color=color.black, scale=0.15, position=(eye_dist, eye_y, eye_z_offset + 0.01))
        self.velocity = Vec3(0,0,0)

    def update(self):
        """Applies velocity, friction, and ensures player stays on the ground and within bounds."""
        dt = time.dt
        if dt == 0: return

        self.position += self.velocity * dt
        # Apply friction
        self.velocity = lerp(self.velocity, Vec3(0,0,0), dt * PHYSICS_CONFIG.get('PLAYER_FRICTION', 1.5))

        if ground:
            self.y = ground.y + self.scale_y / 2
        
        clamp_player_position(self)

class Ball(Entity):
    def __init__(self, position):
        super().__init__(model='sphere', texture='assets/soccer_ball_texture.png', position=position, scale=1.2, collider='sphere')
        self.velocity = Vec3(0,0,0)
        self.gravity = -9.8

    def update(self):
        dt = time.dt
        if dt == 0: return

        if self.velocity.xz.length_squared() > 0:
            distance = self.velocity.xz.length() * dt
            radius = self.scale.x / 2
            rotation_amount = (distance / radius) * (180 / math.pi)
            rotation_axis = self.velocity.cross(Vec3(0, 1, 0)).normalized()
            q = Quat(); q.setFromAxisAngle(rotation_amount, rotation_axis)
            self.quaternion = q * self.quaternion

        self.velocity.y += self.gravity * dt
        self.position += self.velocity * dt

        ground_level = ground.y + self.scale.y/2
        if self.y < ground_level:
            self.y = ground_level
            self.velocity.y *= -0.6

        if abs(self.x) > GAME_CONFIG['FIELD_WIDTH']/2:
            if abs(self.z) > goal_width/2:
                self.velocity.x *= -0.9
                self.x = math.copysign(GAME_CONFIG['FIELD_WIDTH']/2, self.x)
        if abs(self.z) > GAME_CONFIG['FIELD_LENGTH']/2:
            self.velocity.z *= -0.9
            self.z = math.copysign(GAME_CONFIG['FIELD_LENGTH']/2, self.z)

        on_ground = self.y <= ground_level + 0.01
        if on_ground:
            self.velocity.xz = lerp(self.velocity.xz, Vec2(0,0), dt * 1.0)
            if abs(self.velocity.y) < 1: self.velocity.y = 0
        else:
            self.velocity = lerp(self.velocity, Vec3(0,0,0), dt * 0.1)

def clamp_player_position(player):
    """ Ensures a player stays within the field boundaries, accounting for rotation. """
    rotation_rad = math.radians(player.rotation_y)
    cos_theta = abs(math.cos(rotation_rad))
    sin_theta = abs(math.sin(rotation_rad))
    half_x = player.scale_x / 2
    half_z = player.scale_z / 2
    aabb_half_width = half_x * cos_theta + half_z * sin_theta
    aabb_half_depth = half_x * sin_theta + half_z * cos_theta

    player.x = clamp(player.x, -GAME_CONFIG['FIELD_WIDTH']/2 + aabb_half_width, GAME_CONFIG['FIELD_WIDTH']/2 - aabb_half_width)
    player.z = clamp(player.z, -GAME_CONFIG['FIELD_LENGTH']/2 + aabb_half_depth, GAME_CONFIG['FIELD_LENGTH']/2 - aabb_half_depth)

def handle_player_collisions(player1_entity, player2_entity):
    """ Detects and resolves collisions between the two players. """
    if player1_entity.intersects(player2_entity).hit:
        p_pos_1 = player1_entity.position
        p_pos_2 = player2_entity.position

        direction = p_pos_1 - p_pos_2
        direction.y = 0
        dist = direction.length()

        if dist == 0:
            direction = Vec3(random.uniform(-1, 1), 0, random.uniform(-1, 1)).normalized()
        else:
            direction.normalize()

        def get_projected_radius(player, axis):
            rot_rad = math.radians(player.rotation_y)
            x_axis = Vec3(math.cos(rot_rad), 0, -math.sin(rot_rad))
            z_axis = Vec3(math.sin(rot_rad), 0, math.cos(rot_rad))
            radius_x = abs(axis.dot(x_axis)) * (player.scale_x / 2)
            radius_z = abs(axis.dot(z_axis)) * (player.scale_z / 2)
            return radius_x + radius_z

        radius_1 = get_projected_radius(player1_entity, direction)
        radius_2 = get_projected_radius(player2_entity, direction)

        overlap = (radius_1 + radius_2) - dist

        if overlap > 0:
            player1_entity.position += direction * (overlap / 2)
            player2_entity.position -= direction * (overlap / 2)
            clamp_player_position(player1_entity)
            clamp_player_position(player2_entity)

def move_player(player, action):
    action_id = action.item()
    dt = time.dt
    if dt == 0: return

    if action_id == ACTIONS['TURN_LEFT']: # Turn Left
        player.rotation_y -= PHYSICS_CONFIG['PLAYER_TURN_SPEED'] * dt
    elif action_id == ACTIONS['TURN_RIGHT']: # Turn Right
        player.rotation_y += PHYSICS_CONFIG['PLAYER_TURN_SPEED'] * dt
    elif action_id == ACTIONS['ACCELERATE']: # Accelerate Forward
        player.velocity += player.forward * dt * PHYSICS_CONFIG.get('PLAYER_ACCELERATION', 40)
        # Cap speed
        max_speed = PHYSICS_CONFIG.get('PLAYER_MAX_SPEED', 15)
        if player.velocity.length() > max_speed:
            player.velocity = player.velocity.normalized() * max_speed

def handle_ball_kicks(hit_info, ball, agents, prev_ball_dist_to_opp_goals):
    kick_rewards = {agent.team_name: 0 for agent in agents}
    kicker = hit_info.entity
    kicker_agent = next((agent for agent in agents if agent.player == kicker), None)

    if kicker_agent:
        ball.velocity = kicker.forward * PHYSICS_CONFIG['KICK_STRENGTH'] + Vec3(0, PHYSICS_CONFIG['KICK_LIFT'], 0)
        kick_rewards[kicker_agent.team_name] += DQN_CONFIG['REWARD_KICK']
        # Reward for kicking towards opponent's goal
        if distance_xz(ball.position, kicker_agent.opp_goal.position) < prev_ball_dist_to_opp_goals[kicker_agent.team_name]:
            kick_rewards[kicker_agent.team_name] += DQN_CONFIG['REWARD_KICK_TOWARDS_GOAL']

    return kick_rewards

def calculate_base_reward(agent, ball):
    reward = DQN_CONFIG['PENALTY_TIME']
    
    current_dist_to_ball = distance_xz(agent.player.position, ball.position)
    if agent.last_dist_to_ball is not None:
        reward += (agent.last_dist_to_ball - current_dist_to_ball) * DQN_CONFIG['REWARD_MOVE_TO_BALL_SCALE']
    
    agent.last_dist_to_ball = current_dist_to_ball
    return reward 