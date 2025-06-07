from ursina import Text, destroy, Button, color
from config import GAME_CONFIG

class UIManager:
    """ Manages all UI elements like score and timer displays. """
    def __init__(self, start_callback=None):
        if not GAME_CONFIG['SHOULD_RENDER']:
            self.score_display = None
            self.timer_display = None
            self.no_touch_timer_display = None
            self.p1_max_reward_display = None
            self.p2_max_reward_display = None
            self.p1_avg_reward_display = None
            self.p2_avg_reward_display = None
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
        
        self.no_touch_timer_display = Text(
            "00", 
            origin=(0,0), 
            y=0.3, 
            scale=1.1, 
            color=color.orange,
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

    def update_no_touch_timer(self, time_left):
        if not self.no_touch_timer_display: return
        self.no_touch_timer_display.text = f"No Touch: {int(time_left):02}"

    def update_reward_displays(self, agents):
        if not self.p1_max_reward_display: return # Assume others are also None
        self.p1_max_reward_display.text = f"P1 Max Reward: {agents[0].max_reward:.4f}"
        self.p2_max_reward_display.text = f"P2 Max Reward: {agents[1].max_reward:.4f}"
        self.p1_avg_reward_display.text = f"P1 Avg Reward: {agents[0].average_reward:.4f}"
        self.p2_avg_reward_display.text = f"P2 Avg Reward: {agents[1].average_reward:.4f}"

    def destroy_start_button(self):
        if self.start_button:
            destroy(self.start_button)

    def destroy(self):
        """ Destroys the UI elements. """
        if self.score_display:
            destroy(self.score_display)
        if self.timer_display:
            destroy(self.timer_display)
        if self.no_touch_timer_display:
            destroy(self.no_touch_timer_display)
        if self.p1_max_reward_display:
            destroy(self.p1_max_reward_display)
        if self.p2_max_reward_display:
            destroy(self.p2_max_reward_display)
        if self.p1_avg_reward_display:
            destroy(self.p1_avg_reward_display)
        if self.p2_avg_reward_display:
            destroy(self.p2_avg_reward_display)
        self.destroy_start_button() 