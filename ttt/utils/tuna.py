import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch as th


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
        sample_fn,
        train_steps=10000,
        n_eval_eps=50,
        steps_btwn_evals=None
        ):
    steps_btwn_evals = steps_btwn_evals or train_steps // 10

    def train(trial: optuna.Trial):
        kwargs = sample_fn(trial)
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
        train_steps,
        sample_fn,
        n_startup_trials=5,
        n_warmup_steps=10,
        eval_period=1000,
        n_max_trials=100,
        timeout_s=1800,
        n_jobs=1
        ):
    """
    Params
        - name: name of the study. a db file with this name will be saved
        - mkmodel: (kwargs) -> model func
        - mkenv: _ -> gym Env func
        - train_steps: number of steps to train the model in each trial
        - sample_fn: (trial) -> kwargs dict for model creation
        - n_startup_trials: no pruning before this many trials
        - n_warmup_steps: don't prune a trial before this many 'steps'. I think
          a step is an eval interval.
        - eval_period: num training steps between evaluations
        - n_max_trials: stop study after this many trials
        - timeout_s: stop study after this time
        - n_jobs: parallelism. May not work well with GPU models. Try with cpu
    """
    # Set pytorch num threads to 1 for faster training
    th.set_num_threads(1)
    sampler = TPESampler(n_startup_trials=n_startup_trials)
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps
    )
    study = optuna.create_study(
        storage=f"sqlite:///{name}.db",
        study_name=name,
        sampler=sampler,
        pruner=pruner,
        direction="maximize")

    try:
        study.optimize(
            mktrain(mkmodel, mkenv, sample_fn, train_steps, steps_btwn_evals=eval_period),
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

    print("For more details:")
    print(f"uv run optuna-dashboard sqlite:///{name}.db")
