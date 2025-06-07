# --- Game & Simulation Config ---
GAME_CONFIG = {
    'SHOULD_RENDER': True,
    'FIELD_WIDTH': 40,
    'FIELD_LENGTH': 25,
    'GOAL_WIDTH': 20,
    'WALL_HEIGHT': 12,
    'GAME_TIMER_SECONDS': 15,
    'NUM_GAMES_TO_RUN': 100,
    'SAVE_DIR': "models",
}

# --- Physics & Gameplay Config ---
PHYSICS_CONFIG = {
    'PLAYER_ACCELERATION': 20,
    'PLAYER_MAX_SPEED': 13,
    'PLAYER_FRICTION': 1.5,
    'PLAYER_TURN_SPEED': 200,
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

# --- AI Hyperparameters ---
DQN_CONFIG = {
    # Network and State
    # 'ACTION_SIZE': 3, # This is now len(ACTIONS)
    # Training
    'BATCH_SIZE': 128,
    'GAMMA': 0.99,       # Discount factor
    'LR': 1e-4,          # Learning Rate
    'TAU': 0.005,        # Target network soft update rate
    'MEMORY_CAPACITY': 50000,
    'UPDATE_EVERY': 4,              # How often to run the optimization step
    'TARGET_UPDATE_EVERY': 100,     # How often to soft-update the target network
    'CHECKPOINT_EVERY': 50000,
    'GRAD_CLIP': 100,
    # Epsilon-Greedy Exploration
    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'EPS_DECAY': 30000, # Slower decay for more exploration
    # Rewards
    'REWARD_GOAL': 100,
    'REWARD_KICK': 5,
    'REWARD_KICK_TOWARDS_GOAL': 10,
    'REWARD_MOVE_TO_BALL_SCALE': 0.5,
    'REWARD_DEFENSIVE_POS': 0.1,
    'PENALTY_TIME': -0.01,
    'PENALTY_CONCEDE': -100,
} 