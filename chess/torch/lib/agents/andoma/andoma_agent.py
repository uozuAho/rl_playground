from lib import env
from lib.agents.agent import ChessAgent
from lib.agents.andoma.movegeneration import next_move
from lib.agents.mcts import MctsAgent
from lib.michniew import evaluate_board


class AndomaAgent(ChessAgent):
    """Andrew Healey's andoma agent: https://github.com/healeycodes/andoma"""

    def __init__(self, player: env.Player, search_depth=1):
        self.player = player
        self.search_depth = search_depth

    def get_action(self, env):
        assert env.turn == self.player
        return next_move(self.search_depth, env._board, debug=False)


class AndomaMctsAgent(ChessAgent):
    """Similar to Andoma, but use MCTS + michniewski board evaluation instead
    of alpha beta"""

    def __init__(self, player: env.Player, n_sims: int):
        self.player = player
        self._mctsAgent = MctsAgent(
            player, n_sims, valfn=self._valfn, use_valfn_for_expand=True
        )

    def get_action(self, env):
        assert env.turn == self.player
        return self._mctsAgent.get_action(env)

    def _valfn(self, board: env.ChessGame, player: env.Player):
        return evaluate_board(board._board) * player
