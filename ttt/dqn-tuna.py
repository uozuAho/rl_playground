import optuna
import ttt.env
from stable_baselines3 import DQN
from ttt.agents.random import RandomAgent
from ttt.env import TicTacToeEnv
import torch.nn as nn
from utils import tuna


def make_env():
    return TicTacToeEnv(
        opponent=RandomAgent(),
        on_invalid_action=ttt.env.INVALID_ACTION_GAME_OVER)


def sample_params(trial: optuna.Trial):
    net_arches = {
        "32": [32],
        "32_32": [32, 32],
        "64": [64],
        "64_64": [64, 64]
    }

    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    # Display true values (optuna reports the suggested value)
    trial.set_user_attr("gamma_", gamma)
    net_arch = trial.suggest_categorical("net_arch", net_arches.keys())
    net_arch = net_arches[net_arch]
    activation_fn = trial.suggest_categorical("act_fn", ['tanh', 'relu'])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "env": make_env(),
        "policy": "MlpPolicy",
        "batch_size": trial.suggest_int("batch_size", 16, 256),
        "buffer_size": trial.suggest_int("buffer_size", 128, 10000),
        "gamma": gamma,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1, log=True),
        "learning_starts": trial.suggest_int("learning_starts", 10, 10000),
        "target_update_interval": trial.suggest_int("tgt_update_int", 10, 10000),
        "exploration_initial_eps": trial.suggest_float("eps_init", 0.5, 1.0),
        "exploration_final_eps": trial.suggest_float("eps_final", 0.0, 0.2),
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
    }


tuna.run_trials(
    'dqn-ttt',
    mkmodel=lambda kw: DQN(**kw),
    mkenv=make_env,
    train_steps=50000,
    sample_fn=sample_params,
    n_max_trials=100
)
