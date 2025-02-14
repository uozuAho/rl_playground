""" Tabular sarsa learning with afterstate values:

    state,action -> opponent action -> afterstate,next_action
        ^                                                   |
        |                                                   |
        ------------- value update -------------------------

    This is not quite the same as sutton&barto's example, which
    just uses state value estimation (not state,action). I don't
    know how just state is supposed to work, since how can you know
    the subsequent state in order to choose the best action?
"""

import json
import random
import numpy as np

from ttt.env import TicTacToeEnv


class Qtable:
    """ State+action q table"""
    def __init__(self, data=None):
        self._table = data or {}

    def size(self):
        return len(self._table)

    def value(self, state: str, action: int):
        return self._table.get((state, action), 0.0)

    def set_value(self, state: str, action: int, value: float):
        self._table[(state, action)] = value

    def save(self, path):
        with open(path, 'w') as ofile:
            ofile.write(json.dumps(self._table, indent=2))

    def _env2state(self, env: TicTacToeEnv):
        return ''.join(str(x) for x in env.board)


class SarsaAgent:
    def __init__(self, q_table: Qtable | None = None):
        self._q_table = q_table or Qtable()

    @staticmethod
    def load(path):
        with open(path) as infile:
            data = json.loads(infile.read())
        agent = SarsaAgent()
        agent._q_table = Qtable(data)
        return agent

    def get_action(self, env: TicTacToeEnv):
        return greedy_policy(env, self._q_table)

    def save(self, path):
        self._q_table.save(path)

    def train(self,
            env: TicTacToeEnv,
            n_training_episodes,
            min_epsilon=0.001,     # epsilon: exploration rate
            max_epsilon=1.0,
            eps_decay_rate=0.0005, # rate at which exploration drops off
            learning_rate=0.5,
            gamma=0.95,            # discount rate (discount past rewards)
            ep_callback=None
            ):
        for episode in range(n_training_episodes):

            epsilon = min_epsilon + (
                (max_epsilon - min_epsilon) *
                np.exp(-eps_decay_rate * episode))
            env.reset()
            state = env.copy()
            action = next_action = egreedy_policy(env, self._q_table, epsilon)
            done = False

            while not done:
                # todo (maybe): don't update value after exploratory step? (off policy)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = env
                next_action = egreedy_policy(env, self._q_table, epsilon)
                done = terminated or truncated

                current_q = self._q_table.value(envstate(state), action)
                next_q = self._q_table.value(envstate(next_state), next_action)
                new_value = current_q + learning_rate * (reward + gamma * next_q - current_q)
                self._q_table.set_value(envstate(state), action, new_value)

                state = next_state.copy()
                action = next_action

            if ep_callback:
                ep_callback(episode, epsilon)


def make_env(opponent):
    return TicTacToeEnv(opponent=opponent)


def greedy_policy(env: TicTacToeEnv, qtable: Qtable):
    best_value = -999999999.9
    best_action = None
    for a in env.valid_actions():
        value = qtable.value(envstate(env), a)
        if value > best_value:
            best_value = value
            best_action = a
    return best_action


def egreedy_policy(env: TicTacToeEnv, qtable: Qtable, epsilon: float):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(env, qtable)
    else:
        action = env.action_space.sample(mask=env.valid_action_mask())
    return action


def envstate(env: TicTacToeEnv):
    return ''.join(str(x) for x in env.board)
