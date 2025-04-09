import random
import numpy as np

from ttt.agents.agent import TttAgent2
from ttt.agents.sarsa import QaTable
import ttt.env as ttt


class TabQlearnAgent(TttAgent2):
    """
    Tabular q-learning agent. Stores a state-action q table. Trains against
    itself (ie don't supply an opponent).
    """
    def __init__(
            self,
            q_table: QaTable | None = None,
            allow_invalid_actions=False):
        self._q_table = q_table or QaTable()
        self.allow_invalid_actions = allow_invalid_actions

    @staticmethod
    def load(path):
        qtable = QaTable.load(path)
        return TabQlearnAgent(q_table=qtable)

    @staticmethod
    def train_new(opponent: TttAgent2, n_eps: int):
        agent = TabQlearnAgent()
        agent.train(ttt.Env(), opponent, n_eps)
        return agent

    def get_action(self, env: ttt.Env):
        return greedy_policy(env, self._q_table, self.allow_invalid_actions)

    def save(self, path):
        self._q_table.save(path)

    def train(self,
            env: ttt.Env,
            opponent: TttAgent2,
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
                if env.current_player == ttt.O:
                    action = opponent.get_action(env)
                    _, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                if env.current_player == ttt.X:
                    action = egreedy_policy(env, self._q_table, epsilon, self.allow_invalid_actions)
                    _, reward, terminated, truncated, _ = env.step(action)
                    new_state = env
                    done = terminated or truncated

                    maxq = max([self._q_table.value(envstate(new_state), a) for a in range(9)], default=0)
                    current_q = self._q_table.value(envstate(state), action)
                    new_value = current_q + learning_rate * (reward + gamma * maxq - current_q)
                    self._q_table.set_value(envstate(state), action, new_value)

                    state = new_state.copy()

            if ep_callback:
                ep_callback(episode, epsilon)


def greedy_policy(env: ttt.Env, qtable: QaTable, allow_invalid):
    best_value = -999999999.9
    best_action = None
    actions = range(9) if allow_invalid else (env.valid_actions())
    for a in actions:
        value = qtable.value(envstate(env), a)
        if value > best_value:
            best_value = value
            best_action = a
    return best_action


def egreedy_policy(env: ttt.Env, qtable: QaTable, epsilon: float, allow_invalid):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(env, qtable, allow_invalid)
    else:
        action = random.choice(list(env.valid_actions()))
    return action


def envstate(env: ttt.Env):
    return ''.join('x' if c == ttt.X else 'o' if c == ttt.O else '.' for c in env.board)
