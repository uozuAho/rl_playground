import typing as t

import env.connect4 as c4

type Prior = list[float]  # probabiliy distribution of actions
type Value = float  # estimated return from given state
type PV = tuple[Prior, Value]


type BatchEvaluateFunc = t.Callable[[list[c4.GameState]], list[PV]]
