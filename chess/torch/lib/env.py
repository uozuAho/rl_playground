import typing as t

import chess
import numpy as np


type Player = t.Literal[-1, 1]
BLACK: Player = -1  # chess.BLACK = False
WHITE: Player = 1   # chess.WHITE = True

_color_to_player: dict[chess.Color, Player] = {chess.BLACK: BLACK, chess.WHITE: WHITE}


def other_player(player: Player):
    return BLACK if player == WHITE else WHITE


class ChessGame:
    def __init__(
            self,
            fen=None,
            capture_reward_factor=0.0,
            halfmove_limit=None):
        self._board = chess.Board(fen) if fen else chess.Board()
        self.capture_reward_factor = capture_reward_factor
        self.halfmove_limit = halfmove_limit

    @property
    def turn(self) -> Player:
        return _color_to_player[self._board.turn]

    def copy(self):
        """ Return a copy of this state. Expensive! Prefer using the same state
            + undo() where possible
        """
        return ChessGame(self.fen(), self.capture_reward_factor)

    def step(self, move: chess.Move) -> tuple[bool, float]:
        """ Returns: (game_over: bool, reward: float)"""
        self._board.push(move)
        outcome = self._board.outcome()
        reward = 0.0
        captured = self._board.piece_at(move.to_square)
        if captured:
            sign = 1 if captured.color == chess.WHITE else -1
            reward = sign * captured.piece_type * self.capture_reward_factor

        if self._reached_halfmove_limit():
            return True, reward

        if outcome:
            if outcome.winner == chess.WHITE:
                reward = 1.0
            elif outcome.winner == chess.BLACK:
                reward = -1.0
            return True, reward

        return False, reward

    def undo(self):
        self._board.pop()

    def is_game_over(self):
        return self._reached_halfmove_limit() or self._board.is_game_over()

    def _reached_halfmove_limit(self):
        return self.halfmove_limit and len(self._board.move_stack) >= self.halfmove_limit

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

    def state_np(self):
        state = np.zeros(shape=(8, 8, 8), dtype=np.float32)
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self._board.piece_at(i)
            if piece is None:
                continue
            elif piece.symbol().isupper():
                sign = 1  # FEN: upper = white
            else:
                sign = -1 # FEN: lower = black
            layer = piece_layer[piece.symbol()]
            state[layer, row, col] = sign
            state[6, :, :] = 1 / self._board.fullmove_number
        if self._board.turn:
            state[6, 0, :] = 1
        else:
            state[6, 0, :] = -1
        state[7, :, :] = 1
        return state


piece_layer = {}
piece_layer["p"] = 0
piece_layer["r"] = 1
piece_layer["n"] = 2
piece_layer["b"] = 3
piece_layer["q"] = 4
piece_layer["k"] = 5
piece_layer["P"] = 0
piece_layer["R"] = 1
piece_layer["N"] = 2
piece_layer["B"] = 3
piece_layer["Q"] = 4
piece_layer["K"] = 5
