import chess

from env import env
from agents.agent import ChessAgent
from agents.andoma.movegeneration import next_move
from agents.mcts import MctsAgent
from utils.michniew import evaluate_board


class AndomaAgent(ChessAgent):
    """Andrew Healey's andoma agent: https://github.com/healeycodes/andoma"""

    def __init__(self, search_depth=1):
        self.search_depth = search_depth

    def get_action(self, game: env.ChessGame):
        return next_move(self.search_depth, game._board, debug=False)

    def get_actions(self, games: list[env.ChessGame]) -> list[chess.Move]:
        return [self.get_action(g) for g in games]


class AndomaMctsAgent(ChessAgent):
    """Similar to Andoma, but use MCTS + michniewski board evaluation instead
    of alpha beta"""

    def __init__(self, n_sims: int):
        self._mctsAgent = MctsAgent(
            n_sims, valfn=self._valfn, use_valfn_for_expand=True
        )

    def get_action(self, game: env.ChessGame):
        return self._mctsAgent.get_action(game)

    # todo: use parallel mcts

    def _valfn(self, board: env.ChessGame, player: env.Player):
        return evaluate_board(board._board) * player
