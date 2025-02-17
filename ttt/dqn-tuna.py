import ttt.env
from stable_baselines3 import DQN
from ttt.agents.random import RandomAgent
from ttt.env import TicTacToeEnv
from utils import tuna


def auto_tune_hyperparams():
    tuna.run_trials(
        'dqn-ttt',
        mkmodel=lambda kw: DQN(**kw),
        mkenv=lambda: TicTacToeEnv(
            opponent=RandomAgent(),
            on_invalid_action=ttt.env.INVALID_ACTION_GAME_OVER),
        train_steps=10000
    )


auto_tune_hyperparams()
