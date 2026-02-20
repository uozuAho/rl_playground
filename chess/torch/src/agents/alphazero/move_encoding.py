from typing import Protocol

import chess


class Codec(Protocol):
    ACTION_SIZE: int

    def move2int(self, move: chess.Move) -> int:
        pass

    def int2move(self, i: int) -> chess.Move:
        pass


class Codec4096(Codec):
    """Simple linear encoding of all 64x64 from-to combinations"""

    ACTION_SIZE = 4096

    def move2int(self, move: chess.Move):
        f, t = move.from_square, move.to_square
        return f * 64 + t

    def int2move(self, i: int) -> chess.Move:
        # todo: promotions. Eg always promote to queen
        return chess.Move(from_square=i // 64, to_square=i % 64)
