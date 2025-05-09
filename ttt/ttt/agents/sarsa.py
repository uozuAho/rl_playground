""" Tabular sarsa learning with afterstate values:

    state,action -> opponent action -> afterstate,next_action
        ^                                                   |
        |                                                   |
        ------------- value update -------------------------

    This is not quite the same as sutton&barto's example, which
    just uses state value estimation (not state,action). I don't
    know how just state is supposed to work, since how can you know
    the subsequent state in order to choose the best action?

    Only plays well as X. I think the issue with O is that no
    states are saved when it's O's turn during training.
"""

import json
import random
import numpy as np

from ttt.agents.agent import TttAgent
import ttt.env as t3


class QaTable:
    """ State+action q table"""
    def __init__(self, table=None):
        self._table = table or {}

    @staticmethod
    def load(path):
        with open(path) as infile:
            serdict = json.loads(infile.read())
        table = QaTable()
        for k, v in serdict.items():
            state, action = k.split('-')
            table.set_value(state, action, v)
        return table

    def size(self):
        return len(self._table)

    def value(self, state: str, action: int):
        return self._table.get((state, action), 0.0)

    def set_value(self, state: str, action: int, value: float):
        self._table[(state, action)] = value

    def save(self, path):
        with open(path, 'w') as ofile:
            serdict = {f'{k[0]}-{k[1]}': v for k, v in self._table.items()}
            ofile.write(json.dumps(serdict, indent=2))

    def _env2state(self, env: t3.FastEnv):
        return ''.join(str(x) for x in env.board)


class TabSarsaAgent(TttAgent):
    def __init__(
            self,
            q_table: QaTable | None = None,
            allow_invalid_actions=False):
        self._q_table = q_table or QaTable()
        self.allow_invalid_actions = allow_invalid_actions

    @staticmethod
    def load(path):
        qtable = QaTable.load(path)
        return TabSarsaAgent(q_table=qtable)

    @staticmethod
    def train_new(opponent: TttAgent, n_eps):
        env = t3.FastEnv()
        agent = TabSarsaAgent()
        agent.train(env, opponent, n_eps)
        return agent

    def get_action(self, env: t3.Env):
        return greedy_policy(env, self._q_table, self.allow_invalid_actions)

    def save(self, path):
        self._q_table.save(path)

    def train(self,
            env: t3.FastEnv,
            opponent: TttAgent,
            n_training_episodes: int,
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
            action = next_action = egreedy_policy(env, self._q_table, epsilon, self.allow_invalid_actions)
            done = False

            while not done:
                assert env.current_player == t3.X
                _, reward, terminated, truncated, _ = env.step(action)
                if not (terminated or truncated):
                    _, reward, terminated, truncated, _ = env.step(opponent.get_action(env))
                done = terminated or truncated
                next_state = env
                next_action = None if done else egreedy_policy(env, self._q_table, epsilon, self.allow_invalid_actions)

                current_q = self._q_table.value(envstate(state), action)
                next_q = self._q_table.value(envstate(next_state), next_action)
                new_value = current_q + learning_rate * (reward + gamma * next_q - current_q)
                self._q_table.set_value(envstate(state), action, new_value)

                state = next_state.copy()
                action = next_action

            if ep_callback:
                ep_callback(episode, epsilon)


def greedy_policy(env: t3.Env, qtable: QaTable, allow_invalid):
    # assumes qtable has been trained for playing X
    actions = list(range(9)) if allow_invalid else list(env.valid_actions())
    assert actions
    if env.current_player == t3.X:
        return max(actions, key=lambda a: qtable.value(envstate(env), a))
    else:
        return min(actions, key=lambda a: qtable.value(envstate(env), a))


def egreedy_policy(env: t3.Env, qtable: QaTable, epsilon: float, allow_invalid):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(env, qtable, allow_invalid)
    else:
        valid = list(env.valid_actions())
        if valid:
            action = random.choice(list(env.valid_actions()))
        else:
            action = None
    return action


def envstate(env: t3.Env):
    return ''.join(str(x) for x in env.board)
