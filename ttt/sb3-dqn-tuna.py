import os
import optuna
from ttt.agents.random import RandomAgent
from stable_baselines3 import DQN
import torch.nn as nn
from utils import tuna
import ttt.env


def make_env():
    return ttt.env.EnvWithOpponent(
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
    net_arch = net_arches[trial.suggest_categorical("net_arch", list(net_arches.keys()))]
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[trial.suggest_categorical("act_fn", ['tanh', 'relu'])]

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


study_name = 'dqn-ttt'
if os.path.exists(f"{study_name}.db"):
    os.remove(f"{study_name}.db")

train_steps = 50000
# train_steps = 100  # for quick testing

tuna.run_trials(
    study_name,
    mkmodel=lambda kw: DQN(**kw),
    mkenv=make_env,
    train_steps=train_steps,
    sample_fn=sample_params,
    n_max_trials=100
)
