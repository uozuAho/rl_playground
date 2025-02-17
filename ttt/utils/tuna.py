import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from typing import Any, Dict
import torch as th
import torch.nn as nn


def sample_params(trial: optuna.Trial, env: gym.Env) -> Dict[str, Any]:
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ['tiny', 'small'])
    activation_fn = trial.suggest_categorical("act_fn", ['tanh', 'relu'])

    # Display true values
    trial.set_user_attr("gamma_", gamma)

    net_arch = [32, 32] if net_arch == 'tiny' else [64, 64]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "env": env,
        "policy": "MlpPolicy",
        "batch_size": 128,  # todo: parameterise this
        "buffer_size": 10000,  # todo: and this
        "gamma": gamma,
        "learning_rate": learning_rate,
        "learning_starts": 1000, # todo: and this
        "target_update_interval": 1000,  # todo: and this
        "exploration_initial_eps": 1.0, # tiodo
        "exploration_final_eps": 0.1, # toodeooo
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
    }


class TrialEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=True,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def mktrain(
        mkmodel,
        mkenv,
        train_steps=10000,
        n_eval_eps=50,
        steps_btwn_evals=1000
        ):

    def train(trial: optuna.Trial):
        kwargs = sample_params(trial, env=mkenv())
        model = mkmodel(kwargs)
        eval_envs = make_vec_env(mkenv, 5)
        eval_callback = TrialEvalCallback(eval_envs, trial, n_eval_eps, steps_btwn_evals)

        nan_encountered = False
        try:
            model.learn(train_steps, callback=eval_callback)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True
        finally:
            model.env.close()
            eval_envs.close()

        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.last_mean_reward

    return train


def run_trials(
        name,
        mkmodel,
        mkenv,
        n_startup_trials=5,
        n_evaulations=2,
        n_max_trials=10,
        timeout_s=60,
        n_jobs=1
        ):
    """
    Params
        - name: name of the study. a file with this name will be saved
        - mkmodel: (kwargs) -> model func
        - mkenv: _ -> gym Env func
        - n_startup_trials: Stop random sampling after this many trials
        - n_evaulations: Number of evaluations during the training
        - n_max_trials: stop study after this many trials
        - timeout_s: stop study after this time
    """
    # Set pytorch num threads to 1 for faster training
    th.set_num_threads(1)
    sampler = TPESampler(n_startup_trials=n_startup_trials)
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials, n_warmup_steps=n_evaulations // 3
    )
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name="dqn-ttt",
        sampler=sampler,
        pruner=pruner,
        direction="maximize")

    try:
        study.optimize(
            mktrain(mkmodel, mkenv),
            n_trials=n_max_trials,
            n_jobs=n_jobs,
            timeout=timeout_s
        )
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(f"{name}.trials.csv")

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()
