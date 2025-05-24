import typing as t

import chess


type Player = t.Literal[-1, 1]
BLACK: Player = -1  # chess.BLACK = False
WHITE: Player = 1   # chess.WHITE = True

_color_to_player: dict[chess.Color, Player] = {chess.BLACK: BLACK, chess.WHITE: WHITE}


def other_player(player: Player):
    return BLACK if player == WHITE else WHITE


class ChessGame:
    """ Wraps RLC env """
    def __init__(self, fen=None, capture_reward_factor=0.01):
        self._board = chess.Board(fen) if fen else chess.Board()
        self.capture_reward_factor = capture_reward_factor

    @property
    def turn(self) -> Player:
        return _color_to_player[self._board.turn]

    def copy(self):
        return ChessGame(self.fen(), self.capture_reward_factor)

    def step(self, move: chess.Move) -> tuple[bool, float]:
        self._board.push(move)
        outcome = self._board.outcome()
        reward = 0.0
        done = False
        if outcome:
            done = True
            if outcome.winner == chess.WHITE:
                reward = 1.0
            elif outcome.winner == chess.BLACK:
                reward = -1.0
        return done, reward

    def undo(self):
        self._board.pop()

    def is_game_over(self):
        return self._board.is_game_over()

    def legal_moves(self):
        return self._board.generate_legal_moves()

    def fen(self):
        return self._board.fen()

    def winner(self) -> Player | None:
        outcome = self._board.outcome()
        if not outcome:
            return None
        if not outcome.winner:
            return None
        return _color_to_player[outcome.winner]
