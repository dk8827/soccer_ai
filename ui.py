from ursina import Text, destroy
from config import GAME_CONFIG

class UIManager:
    """ Manages all UI elements like score and timer displays. """
    def __init__(self):
        if not GAME_CONFIG['SHOULD_RENDER']:
            self.score_display = None
            self.timer_display = None
            return

        self.score_display = Text(origin=(0,0), y=0.4, scale=1.5, background=True)
        self.timer_display = Text("00:00", origin=(0,0), y=0.35, scale=1.2, background=True)
        self.update_score({'player1': 0, 'player2': 0}) # Initialize score text

    def update_score(self, score):
        if not self.score_display: return
        self.score_display.text = f"<orange>{score['player1']}<default> - <azure>{score['player2']}"

    def update_timer(self, time_left):
        if not self.timer_display: return
        mins, secs = divmod(time_left, 60)
        self.timer_display.text = f"Time: {int(mins):02}:{int(secs):02}"

    def destroy(self):
        """ Destroys the UI elements. """
        if self.score_display:
            destroy(self.score_display)
        if self.timer_display:
            destroy(self.timer_display) 