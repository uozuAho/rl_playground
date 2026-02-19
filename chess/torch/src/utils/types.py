import typing as t

import chess

from env import env

type Prior = list[float]  # probabiliy distribution of actions
type Value = float  # estimated return from given state
type PV = tuple[Prior, Value]

type MoveProbs = dict[chess.Move, float]
type MPV = tuple[MoveProbs, Value]

type BatchEvaluateFunc = t.Callable[[list[env.ChessGame]], list[MPV]]
