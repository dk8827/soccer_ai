from ursina import *
import sys
import math
from panda3d.core import Quat

def cross(v1, v2):
    return Vec3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    )

# ----------------- GAME SETUP -----------------
app = Ursina()

# Window and Scene setup
window.title = 'Cube Soccer 3D'
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False
window.fps_counter.enabled = True

# Define field dimensions for easy adjustments
FIELD_WIDTH = 40
FIELD_LENGTH = 25

# Game state variables
score = {'orange': 0, 'blue': 0}
game_over = False
GAME_TIMER_SECONDS = 180 # 3 minutes
time_left = GAME_TIMER_SECONDS

# ----------------- ENVIRONMENT -----------------
# The ground/field
ground = Entity(
    model='quad',
    texture='assets/grid_texture.png',
    scale=(FIELD_WIDTH, FIELD_LENGTH),
    rotation_x=90,
    collider='box',
    y=-0.5
)

# The surrounding walls
wall_texture = 'assets/wall_texture.png'
wall_height = 12
goal_width = 20 # Must match goal's width

# Top and bottom walls
Entity(model='cube', texture=wall_texture, scale=(FIELD_WIDTH + 2, wall_height, 1), position=(0, wall_height/2 - 0.5, FIELD_LENGTH/2 + 0.5), collider='box')
Entity(model='cube', scale=(FIELD_WIDTH + 2, wall_height, 1), position=(0, wall_height/2 - 0.5, -FIELD_LENGTH/2 - 0.5), collider='box', visible=False)

# Side walls (with goal openings)
wall_segment_length = (FIELD_LENGTH - goal_width) / 2
wall_z_offset = goal_width/2 + wall_segment_length/2
# Left wall segments
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(-FIELD_WIDTH/2 - 0.5, wall_height/2 - 0.5, -wall_z_offset), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(-FIELD_WIDTH/2 - 0.5, wall_height/2 - 0.5, wall_z_offset), collider='box')
# Right wall segments
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(FIELD_WIDTH/2 + 0.5, wall_height/2 - 0.5, -wall_z_offset), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, wall_height, wall_segment_length), position=(FIELD_WIDTH/2 + 0.5, wall_height/2 - 0.5, wall_z_offset), collider='box')

# Wall segments above goals
Entity(model='cube', texture=wall_texture, scale=(1, 1.25, goal_width), position=(-FIELD_WIDTH/2 - 0.5, 10.875, 0), collider='box')
Entity(model='cube', texture=wall_texture, scale=(1, 1.25, goal_width), position=(FIELD_WIDTH/2 + 0.5, 10.875, 0), collider='box')

# ----------------- GOAL CLASS -----------------
class Goal(Entity):
    def __init__(self, clr, **kwargs):
        super().__init__(**kwargs)
        w, h = 20, 10  # width, height
        recess_depth = 0.5
        
        # The frame consists of Entities parented to this Goal Entity
        # This makes rotating the entire goal easy
        frame_color = color.orange if clr == 'orange' else color.blue
        Entity(parent=self, model='cube', collider='box', scale=(w, .5, .5), position=(0, h, recess_depth), color=frame_color)
        Entity(parent=self, model='cube', collider='box', scale=(.5, h, .5), position=(-w/2, h/2, recess_depth), color=frame_color)
        Entity(parent=self, model='cube', collider='box', scale=(.5, h, .5), position=(w/2, h/2, recess_depth), color=frame_color)

        # Net behind the frame
        Entity(parent=self, model='quad', texture='assets/net_texture.png', double_sided=True,
               texture_scale=(w/10, h/5),
               scale=(w,h), position=(0, h/2, recess_depth + 0.05))

        # Invisible trigger for detecting goals. It's slightly inside the frame.
        self.trigger = Entity(parent=self, model='cube', collider='box',
                              scale=(w-0.5, h-0.5, 1),
                              position=(0, h/2, recess_depth + 0.5),
                              visible=False)

# Create the two goals
orange_goal = Goal(clr='orange', position=(-FIELD_WIDTH/2 - 0.5, 0, 0), rotation_y=-90)
blue_goal = Goal(clr='blue', position=(FIELD_WIDTH/2 + 0.5, 0, 0), rotation_y=90)

# ----------------- PLAYER AND BALL CLASSES -----------------
class Player(Entity):
    def __init__(self, position, clr, controls):
        super().__init__(
            model='cube',
            color=clr,
            position=position,
            scale=1.8,
            collider='box'
        )
        self.speed = 7
        self.turn_speed = 120
        self.controls = controls
        
        # Cute googly eyes
        eye_dist = 0.25
        eye_y = 0.2
        eye_z_offset = 0.51
        Entity(parent=self, model='sphere', color=color.white, scale=0.3, position=(-eye_dist, eye_y, eye_z_offset))
        Entity(parent=self, model='sphere', color=color.black, scale=0.15, position=(-eye_dist, eye_y, eye_z_offset + 0.01))
        Entity(parent=self, model='sphere', color=color.white, scale=0.3, position=(eye_dist, eye_y, eye_z_offset))
        Entity(parent=self, model='sphere', color=color.black, scale=0.15, position=(eye_dist, eye_y, eye_z_offset + 0.01))

    def update(self):
        if held_keys[self.controls['fwd']]:
            self.position += self.forward * time.dt * self.speed
        if held_keys[self.controls['left']]:
            self.rotation_y -= self.turn_speed * time.dt
        if held_keys[self.controls['right']]:
            self.rotation_y += self.turn_speed * time.dt
            
        # Keep player within the field bounds
        self.x = clamp(self.x, -FIELD_WIDTH/2, FIELD_WIDTH/2)
        self.z = clamp(self.z, -FIELD_LENGTH/2, FIELD_LENGTH/2)


class AiPlayer(Entity):
    def __init__(self, position, clr, target_ball):
        super().__init__(model='cube', color=clr, position=position, scale=1.8, collider='box')
        self.speed = 6
        self.turn_speed = 100
        self.target = target_ball
        self.goal_pos = blue_goal.position
        self.defensive_x = FIELD_WIDTH / 4

    def update(self):
        # AI Logic: Stay between ball and goal, then attack when ball is on its half
        dist_to_ball = distance_xz(self, self.target)
        
        # If ball is on AI's side of the field, attack it
        if self.target.x > 0:
            target_pos = self.target.position
        # If ball is on human's side, return to a defensive position
        else:
            target_pos = Vec3(self.defensive_x, 0, self.target.z * 0.5)

        # Turn towards the target position
        self.look_at_2d(target_pos, 'y')
        
        # Move forward, but more slowly if just turning
        if self.rotation_y - self.world_rotation_y < 10:
            self.position += self.forward * time.dt * self.speed
            
        # Keep player within the field bounds
        self.x = clamp(self.x, -FIELD_WIDTH/2, FIELD_WIDTH/2)
        self.z = clamp(self.z, -FIELD_LENGTH/2, FIELD_LENGTH/2)


class Ball(Entity):
    def __init__(self, position):
        super().__init__(
            model='sphere',
            texture='assets/soccer_ball_texture.png',
            position=position,
            scale=1.2,
            collider='sphere'
        )
        self.velocity = Vec3(0,0,0)

    def update(self):
        # Ball rolling physics
        if self.velocity.length_squared() > 0:
            # Rotate based on movement
            distance = self.velocity.length() * time.dt
            radius = self.scale.x / 2
            rotation_amount = (distance / radius) * (180 / math.pi)
            rotation_axis = cross(self.velocity, Vec3(0, 1, 0)).normalized()
            
            q = Quat()
            q.setFromAxisAngle(rotation_amount, rotation_axis)
            self.quaternion = q * self.quaternion

        # Apply velocity and simple friction
        self.position += self.velocity * time.dt
        self.velocity = lerp(self.velocity, Vec3(0,0,0), time.dt * 0.5)

        # Bounce off side walls
        if abs(self.x) > FIELD_WIDTH/2:
            self.velocity.x *= -0.9
            self.x = math.copysign(FIELD_WIDTH/2, self.x)
        # Bounce off end walls (if it doesn't go in the goal)
        if abs(self.z) > FIELD_LENGTH/2:
            self.velocity.z *= -0.9
            self.z = math.copysign(FIELD_LENGTH/2, self.z)

# Create the players and the ball
human_player = Player(position=(-10, 0, 0), clr=color.orange, controls={'fwd':'w', 'left':'a', 'right':'d'})
human_player.rotation_y = 90
ball = Ball(position=(0,0,0))
ai_player = AiPlayer(position=(10, 0, 0), clr=color.blue, target_ball=ball)
ai_player.rotation_y = -90

# ----------------- UI (SCORE & TIMER) -----------------
# Main score display in the center
score_display = Text(origin=(0,0), y=0.4, scale=1.5, background=True)
# Timer display at the top
timer_display = Text("03:00", origin=(0, -21), scale=1.5, background=True)
# Side scoreboards mounted on the walls
left_score_board = Entity(parent=scene, model='quad', position=(-FIELD_WIDTH/2 - 0.49, 4, 0), scale=(6, 3), rotation=(0,90,0), color=color.black)
left_score_text = Text("0", parent=left_score_board, scale=10, origin=(0,0), color=color.white)
right_score_board = Entity(parent=scene, model='quad', position=(FIELD_WIDTH/2 + 0.49, 4, 0), scale=(6, 3), rotation=(0,-90,0), color=color.black)
right_score_text = Text("0", parent=right_score_board, scale=10, origin=(0,0), color=color.white)
# Game over message
game_over_text = Text("", origin=(0,0), scale=3, background=True)

def update_score_ui():
    score_display.text = f"The score is <orange>{score['orange']}<default> - <azure>{score['blue']}"
    left_score_text.text = str(score['orange'])
    right_score_text.text = str(score['blue'])

# ----------------- CAMERA AND GAME LOGIC -----------------
# Position the camera to match the image
camera.position = (0, 40, -40)
camera.rotation = (45, 0, 0)

def input(key):
    # Mouse wheel controls camera zoom
    if key == 'scroll up':
        camera.position += camera.forward
    if key == 'scroll down':
        camera.position -= camera.forward
    if key == 'escape':
        sys.exit()

def reset_positions():
    """Resets players and ball to starting positions after a goal."""
    human_player.position = (-10, 0, 0)
    human_player.rotation = (0, 90, 0)
    ai_player.position = (10, 0, 0)
    ai_player.rotation = (0, -90, 0)
    ball.position = (0, 0, 0)
    ball.velocity = Vec3(0,0,0)

def update():
    """Main game loop, called every frame."""
    global time_left, game_over
    
    if game_over:
        return

    # --- TIMER LOGIC ---
    time_left -= time.dt
    if time_left <= 0:
        time_left = 0
        game_over = True
        winner = "Orange" if score['orange'] > score['blue'] else "Blue" if score['blue'] > score['orange'] else "Nobody"
        tie_text = "It's a tie!"
        game_over_text.text = f"GAME OVER\n{winner} wins!" if winner != "Nobody" else f"GAME OVER\n{tie_text}"
        game_over_text.color = color.orange if winner == "Orange" else color.azure if winner == "Blue" else color.white
        invoke(sys.exit, delay=5)
    
    mins, secs = divmod(time_left, 60)
    timer_display.text = f"{int(mins):02}:{int(secs):02}"

    # --- COLLISION LOGIC ---
    # Player bumps ball
    hit_info = ball.intersects()
    if hit_info.hit and hit_info.entity in (human_player, ai_player):
        player = hit_info.entity
        # Apply force to the ball based on the player's forward direction
        ball.velocity = player.forward * 15

    # Ball hits goal trigger
    if ball.intersects(blue_goal.trigger).hit:
        score['orange'] += 1
        update_score_ui()
        reset_positions()
    
    if ball.intersects(orange_goal.trigger).hit:
        score['blue'] += 1
        update_score_ui()
        reset_positions()


# Initial call to set up UI
update_score_ui()
app.run()