# --- Game & Simulation Config ---
GAME_CONFIG = {
    'SHOULD_RENDER': True,
    'FIELD_WIDTH': 40,
    'FIELD_LENGTH': 25,
    'GOAL_WIDTH': 20,
    'WALL_HEIGHT': 12,
    'GAME_TIMER_SECONDS': 40,
    'NO_TOUCH_TIMEOUT': 10,
    'NUM_GAMES_TO_RUN': 2000,
    'SAVE_DIR': "models",
}

# --- Physics & Gameplay Config ---
PHYSICS_CONFIG = {
    'PLAYER_ACCELERATION': 20,
    'PLAYER_MAX_SPEED': 13,
    'PLAYER_FRICTION': 1.5,
    'PLAYER_TURN_SPEED': 200,
    'KICK_STRENGTH': 10,
    'KICK_VELOCITY_BONUS': 0.3, # Multiplier for player's forward speed on kick
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

# --- AI Hyperparameters ---
DQN_CONFIG = {
    # Network and State
    'HIDDEN_LAYER_SIZE': 128,
    # Training
    'BATCH_SIZE': 128,
    'GAMMA': 0.99,       # Discount factor
    'LR': 1e-4,          # Learning Rate
    'TAU': 0.005,        # Target network soft update rate
    'MEMORY_CAPACITY': 50000,
    'UPDATE_EVERY': 4,              # How often to run the optimization step
    'TARGET_UPDATE_EVERY': 1000,     # How often to soft-update the target network
    'CHECKPOINT_EVERY': 50000,
    'GRAD_CLIP': 100,
    # Parameter Space Noise
    'NOISE_ENABLED': True,
    'NOISE_SCALE_START': 0.05,
    'NOISE_SCALE_END': 0.01,
    'NOISE_SCALE_DECAY': 100000,
    # Rewards
    'REWARD_GOAL': 1,
    'REWARD_BALL_PROXIMITY_SCALE': 1,   # Scales reward for being close to the ball (inversely proportional to distance)
    'PENALTY_CONCEDE': -1,
    'PENALTY_STATIONARY': -0.5,           # Penalty for not moving over a window of steps
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