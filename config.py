# --- Game & Simulation Config ---
GAME_CONFIG = {
    'SHOULD_RENDER': True,
    'FIELD_WIDTH': 40,
    'FIELD_LENGTH': 25,
    'GOAL_WIDTH': 20,
    'WALL_HEIGHT': 12,
    'GAME_TIMER_SECONDS': 40,
    'NUM_GAMES_TO_RUN': 2000,
    'SAVE_DIR': "models",
}

# --- Physics & Gameplay Config ---
PHYSICS_CONFIG = {
    'PLAYER_ACCELERATION': 20,
    'PLAYER_MAX_SPEED': 15,
    'PLAYER_FRICTION': 1.5,
    'PLAYER_TURN_SPEED': 250,
    'KICK_STRENGTH': 10,
    'KICK_VELOCITY_BONUS': 0.5, # Multiplier for player's forward speed on kick
    'KICK_LIFT': 6,
    'BALL_BOUNCINESS': 0.6,
    'BALL_WALL_BOUNCINESS': 0.9,
    'BALL_GROUND_FRICTION': 1.0,
    'BALL_AIR_FRICTION': 0.1,
    'BALL_REST_VELOCITY_THRESHOLD': 1.0,
}

# --- AI Action Definitions ---
ACTIONS = {
    'TURN_LEFT': 0,
    'TURN_RIGHT': 1,
    'ACCELERATE': 2,
}

# --- AI Action Definitions (Macros) ---
# Each macro is a tuple: (primitive_action_id, duration_in_frames)
MACRO_ACTIONS = {
    0: (ACTIONS['TURN_LEFT'], 1),      # Turn Left (~2°)
    1: (ACTIONS['TURN_RIGHT'], 1),     # Turn Right (~2°)
    2: (ACTIONS['TURN_LEFT'], 5),      # Turn Left (~10°)
    3: (ACTIONS['TURN_RIGHT'], 5),     # Turn Right (~10°)
    4: (ACTIONS['TURN_LEFT'], 15),     # Turn Left (~30°)
    5: (ACTIONS['TURN_RIGHT'], 15),    # Turn Right (~30°)
    6: (ACTIONS['TURN_LEFT'], 30),     # Turn Left (~60°)
    7: (ACTIONS['TURN_RIGHT'], 30),    # Turn Right (~60°)
    8: (ACTIONS['ACCELERATE'], 1),      # Short Burst
    9: (ACTIONS['ACCELERATE'], 5),     # Long Burst
}

# --- AI Hyperparameters ---
DQN_CONFIG = {
    # Network and State
    'HIDDEN_LAYER_SIZE': 128,
    # Training
    'BATCH_SIZE': 128,
    'GAMMA': 0.99,       # Discount factor
    'LR': 1e-4,          # Learning Rate
    'TAU': 0.005,        # Target network soft update rate
    'MEMORY_CAPACITY': 100000,
    'UPDATE_EVERY': 4,              # How often to run the optimization step
    'TARGET_UPDATE_EVERY': 20,     # How often to soft-update the target network
    'CHECKPOINT_EVERY': 50000,
    'GRAD_CLIP': 10,
    # Parameter Space Noise
    'NOISE_ENABLED': True,
    'NOISE_SCALE_START': 0.05,
    'NOISE_SCALE_END': 0.0001,
    'NOISE_SCALE_DECAY': 100000,
    # Rewards
    'REWARD_GOAL': 10,
    'REWARD_KICK': 0.5,                   #  Reward for making contact with the ball
    'REWARD_BALL_PROXIMITY_SCALE': 0.03,   # Scales reward for being close to the ball (inversely proportional to distance)
    'REWARD_FACING_BALL_SCALE': 0.05,    # Scales reward for facing the ball
    'PENALTY_CONCEDE': -10,
    'PENALTY_SELF_GOAL': -15,             # Harsher penalty for scoring on oneself
    'PENALTY_STATIONARY': -0.1,           # Penalty for not moving over a window of steps
    'STATIONARY_WINDOW': 200,             # Number of steps to check for inactivity
    'STATIONARY_THRESHOLD': 2.0,          # Max distance moved in the window to be considered stationary
}

# --- Curriculum Learning Config ---
CURRICULUM_CONFIG = {
    'ENABLED': True,
    'BALL_Z_RANGE_START': 0.1, # Start with a very small random z range
    'BALL_Z_RANGE_END_FACTOR': 2.5, # FIELD_LENGTH / THIS_VALUE = max z offset
    'SCHEDULE_FRAMES': 500000, # Number of frames to reach the full range
} 