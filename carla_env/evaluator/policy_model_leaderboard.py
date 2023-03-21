
from leaderboard.leaderboard_evaluator_local import main as leaderboard_evaluator_main


class Evaluator:
    def __init__(
        self,
        environment,
        leaderboard,
        device,
    ):

        self.environment = environment
        self.leaderboard = leaderboard
        self.device = device

    def evaluate(self, run):

        leaderboard_evaluator_main(
            self.leaderboard, self.device, self.environment
        )
