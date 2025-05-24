from RLC.real_chess.environment import Board  # type: ignore
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
        self._board = Board(None, fen, capture_reward_factor)

    @property
    def turn(self) -> Player:
        return _color_to_player[self._board.board.turn]

    def copy(self):
        return ChessGame(self._board.board.fen(), self._board.capture_reward_factor)

    def step(self, move: chess.Move) -> tuple[bool, float]:
        return self._board.step(move)

    def undo(self):
        self._board.board.pop()
        self._board.init_layer_board()

    def is_game_over(self):
        return self._board.board.is_game_over()

    def legal_moves(self):
        return self._board.board.generate_legal_moves()

    def fen(self):
        return self._board.board.fen()

    def winner(self) -> Player | None:
        outcome = self._board.board.outcome()
        if not outcome:
            return None
        if not outcome.winner:
            return None
        return _color_to_player[outcome.winner]
