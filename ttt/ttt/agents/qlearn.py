import json
import random
import numpy as np

from ttt.agents.agent import TttAgent
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


class QlearnAgent(TttAgent):
    """
    Tabular q-learning agent. Stores a state-action q table. Trains against
    itself (ie don't supply an opponent).
    """
    def __init__(
            self,
            q_table: Qtable | None = None,
            allow_invalid_actions=False):
        self._q_table = q_table or Qtable()
        self.allow_invalid_actions = allow_invalid_actions

    @staticmethod
    def load(path):
        with open(path) as infile:
            data = json.loads(infile.read())
        agent = QlearnAgent()
        agent._q_table = Qtable(data)
        return agent

    def get_action(self, env: TicTacToeEnv):
        return greedy_policy(env, self._q_table, self.allow_invalid_actions)

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
            done = False

            while not done:
                action = egreedy_policy(env, self._q_table, epsilon, self.allow_invalid_actions)
                new_state, reward, terminated, truncated, _ = env.step(action)
                new_state = env
                done = terminated or truncated

                maxq = max([self._q_table.value(envstate(new_state), a) for a in range(9)], default=0)
                current_q = self._q_table.value(envstate(state), action)
                new_value = current_q + learning_rate * (reward + gamma * maxq - current_q)
                self._q_table.set_value(envstate(state), action, new_value)

                state = new_state.copy()

            if ep_callback:
                ep_callback(episode, epsilon)


def greedy_policy(env: TicTacToeEnv, qtable: Qtable, allow_invalid):
    best_value = -999999999.9
    best_action = None
    actions = range(9) if allow_invalid else (env.valid_actions())
    for a in actions:
        value = qtable.value(envstate(env), a)
        if value > best_value:
            best_value = value
            best_action = a
    return best_action


def egreedy_policy(env: TicTacToeEnv, qtable: Qtable, epsilon: float, allow_invalid):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(env, qtable, allow_invalid)
    else:
        mask = np.ones(9, dtype=np.int8) if allow_invalid else env.valid_action_mask()
        action = env.action_space.sample(mask=mask)
    return action


def envstate(env: TicTacToeEnv):
    return ''.join(str(x) for x in env.board)
