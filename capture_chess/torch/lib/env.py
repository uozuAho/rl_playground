from RLC.capture_chess.environment import Board  # type: ignore
import chess


class CaptureChess:
    def __init__(self, action_limit: int):
        self._board = Board()
        self._action_limit = action_limit

    @property
    def board(self):
        return self._board.board

    @property
    def layer_board(self):
        return self._board.layer_board

    def get_random_action(self):
        return self._board.get_random_action()

    def reset(self):
        return self._board.reset()

    def project_legal_moves(self):
        return self._board.project_legal_moves()

    def step(self, action: chess.Move):
        done, reward = self._board.step(action)

        # no pawn promotion reward:
        if action.promotion:
            val = (
                3
                if action.promotion in [chess.KNIGHT, chess.BISHOP]
                else 5
                if action.promotion == chess.ROOK
                else 9
            )
            reward = reward + 1 - val

        # limit game length
        if self._board.board.fullmove_number == self._action_limit:
            done = True

        return done, reward
