import typing as t

from env import env

type Prior = list[float]  # probabiliy distribution of actions
type Value = float  # estimated return from given state
type PV = tuple[Prior, Value]


type BatchEvaluateFunc = t.Callable[[list[env.ChessGame]], list[PV]]
