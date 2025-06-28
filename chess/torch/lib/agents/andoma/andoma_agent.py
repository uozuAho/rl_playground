from lib import env
from lib.agents.agent import ChessAgent
from lib.agents.andoma.movegeneration import next_move


class AndomaAgent(ChessAgent):
    def __init__(self, player: env.Player, search_depth=1):
        self.player = player
        self.search_depth = search_depth

    def get_action(self, env):
        assert env.turn == self.player
        return next_move(self.search_depth, env._board, debug=False)
