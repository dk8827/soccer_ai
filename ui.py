from ursina import Text, destroy, Button, color, Sequence, Wait, Func, curve
from config import GAME_CONFIG

class UIManager:
    """ Manages all UI elements like score and timer displays. """
    def __init__(self, start_callback=None):
        if not GAME_CONFIG['SHOULD_RENDER']:
            self.score_display = None
            self.timer_display = None
            self.p1_max_reward_display = None
            self.p2_max_reward_display = None
            self.p1_avg_reward_display = None
            self.p2_avg_reward_display = None
            self.game_info_display = None
            self.start_button = None
            return

        # Clean text displays without backgrounds
        self.score_display = Text(
            origin=(0,0), 
            y=0.42, 
            scale=2.0, 
            color=color.white,
            background=False
        )
        
        self.timer_display = Text(
            "00:00", 
            origin=(0,0), 
            y=0.36, 
            scale=1.4, 
            color=color.yellow,
            background=False
        )
        
        self.p1_max_reward_display = Text(
            "P1 Max Reward: 0.0",
            origin=(-.5, 0),
            position=(-0.85, 0.45),
            scale=1.0,
            color=color.orange,
            background=False
        )

        self.p2_max_reward_display = Text(
            "P2 Max Reward: 0.0",
            origin=(.5, 0),
            position=(0.85, 0.45),
            scale=1.0,
            color=color.azure,
            background=False
        )
        
        self.p1_avg_reward_display = Text(
            "P1 Avg Reward: 0.0",
            origin=(-.5, 0),
            position=(-0.85, 0.40),
            scale=1.0,
            color=color.orange,
            background=False
        )

        self.p2_avg_reward_display = Text(
            "P2 Avg Reward: 0.0",
            origin=(.5, 0),
            position=(0.85, 0.40),
            scale=1.0,
            color=color.azure,
            background=False
        )

        self.game_info_display = Text(
            "Game: 0 | Frames: 0",
            origin=(0, 0),
            position=(0, -0.45),
            scale=1.0,
            color=color.white,
            background=False
        )

        self.update_score({'player1': 0, 'player2': 0}) # Initialize score text

        # Improved start button styling
        self.start_button = Button(
            text='Start Game', 
            color=color.green, 
            scale=(0.2, 0.06), 
            y=0,
            text_color=color.white
        )
        self.start_button.on_click = start_callback

    def update_score(self, score):
        if not self.score_display: return
        self.score_display.text = f"<orange>{score['player1']}<default> - <azure>{score['player2']}"

    def update_timer(self, time_left):
        if not self.timer_display: return
        mins, secs = divmod(time_left, 60)
        self.timer_display.text = f"Time: {int(mins):02}:{int(secs):02}"

    def update_reward_displays(self, agents):
        if not self.p1_max_reward_display: return # Assume others are also None
        self.p1_max_reward_display.text = f"P1 Max Reward: {agents[0].max_reward:.4f}"
        self.p2_max_reward_display.text = f"P2 Max Reward: {agents[1].max_reward:.4f}"
        self.p1_avg_reward_display.text = f"P1 Avg Reward: {agents[0].average_reward:.4f}"
        self.p2_avg_reward_display.text = f"P2 Avg Reward: {agents[1].average_reward:.4f}"

    def update_game_info(self, game_number, total_frames):
        if not self.game_info_display: return
        self.game_info_display.text = f"Game: {game_number+1} | Frames: {total_frames}"

    def destroy_start_button(self):
        if self.start_button:
            destroy(self.start_button)

    def destroy(self):
        """ Destroys the UI elements. """
        if self.score_display:
            destroy(self.score_display)
        if self.timer_display:
            destroy(self.timer_display)
        if self.p1_max_reward_display:
            destroy(self.p1_max_reward_display)
        if self.p2_max_reward_display:
            destroy(self.p2_max_reward_display)
        if self.p1_avg_reward_display:
            destroy(self.p1_avg_reward_display)
        if self.p2_avg_reward_display:
            destroy(self.p2_avg_reward_display)
        if self.game_info_display:
            destroy(self.game_info_display)
        self.destroy_start_button()

    def flash_goal_scored(self, scoring_team_name, on_complete):
        if not GAME_CONFIG['SHOULD_RENDER']:
            on_complete()
            return
            
        goal_text = Text(
            text=f"GOAL! {scoring_team_name}",
            origin=(0,0),
            scale=4,
            color=color.gold,
            background=True
        )
        
        # Create a sequence to show, hold, then hide the text
        goal_text.animate_scale(goal_text.scale * 1.2, duration=0.2)
        seq = Sequence(
            Wait(1.5),
            Func(lambda: destroy(goal_text)),
            Func(on_complete)
        )
        seq.start() 