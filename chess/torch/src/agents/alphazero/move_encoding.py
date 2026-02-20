from typing import Protocol

import chess

from env.env import ChessGame
from utils.types import Prior, MoveProbs


class Codec(Protocol):
    ACTION_SIZE: int

    def move2int(self, move: chess.Move) -> int:
        pass

    def int2move(self, i: int) -> chess.Move:
        pass

    def prior2dict(self, probs: Prior, state: ChessGame) -> MoveProbs:
        """convert a probability distribution to a dict of legal move probabilities"""
        pass

    def dict2prior(self, probs: MoveProbs) -> Prior:
        pass

    def validmask(self, state: ChessGame) -> list[bool]:
        pass


class Codec4096(Codec):
    """Simple linear encoding of all 64x64 from-to combinations"""

    ACTION_SIZE = 4096

    def move2int(self, move: chess.Move) -> int:
        f, t = move.from_square, move.to_square
        return f * 64 + t

    def int2move(self, i: int) -> chess.Move:
        # todo: promotions. Eg always promote to queen
        return chess.Move(from_square=i // 64, to_square=i % 64)

    def prior2dict(self, probs: Prior, state: ChessGame) -> MoveProbs:
        legal_moves = list(state.legal_moves())
        mp = {lm: probs[self.move2int(lm)] for lm in legal_moves}
        # ensure a valid probability distribution
        psum = sum(mp.values())
        for m in mp:
            mp[m] = mp[m] / psum
        return mp

    def dict2prior(self, probs: MoveProbs) -> Prior:
        prior = [0] * self.ACTION_SIZE
        for move, prob in probs.items():
            prior[self.move2int(move)] = prob
        return prior

    def validmask(self, state: ChessGame):
        mask = [False] * self.ACTION_SIZE
        for move in state.legal_moves():
            mask[self.move2int(move)] = True
        return mask
