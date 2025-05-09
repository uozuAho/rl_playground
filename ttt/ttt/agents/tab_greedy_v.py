import json
import random
import numpy as np
import typing as t

from ttt.agents.agent import TttAgent
import ttt.env as t3


class Qtable:
    """ State q table"""
    def __init__(self, data=None):
        self._table = data or {}

    def size(self):
        return len(self._table)

    def value(self, state: str):
        return self._table.get(state, 0.0)

    def set_value(self, state: str, value: float):
        self._table[state] = value

    def values(self) -> t.Iterable[t.Tuple[str, float]]:
        for state, val in self._table.items():
            yield state, val

    def save(self, path):
        with open(path, 'w') as ofile:
            ofile.write(json.dumps(self._table, indent=2))


class TabGreedyVAgent(TttAgent):
    """
    Tabular greedy value learning agent. Stores a state q table. Trains against
    a random opponent (ie don't supply an opponent).
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
        agent = TabGreedyVAgent()
        agent._q_table = Qtable(data)
        return agent

    @staticmethod
    def train_new(n_eps: int):
        agent = TabGreedyVAgent()
        agent.train(t3.FastEnv(), n_eps)
        return agent

    def get_action(self, env: t3.Env):
        return greedy_policy(env, self._q_table)

    def action_values(self, board_str: str):
        """ For debugging """
        env = t3.FastEnv.from_str(board_str)
        values = {}
        for a in env.valid_actions():
            tempenv = env.copy()
            tempenv.step(a)
            values[a] = self._q_table.value(envstate(tempenv))
        return values

    def save(self, path):
        self._q_table.save(path)

    def train(self,
            env: t3.Env,
            n_training_episodes,
            min_epsilon=0.001,     # epsilon: exploration rate
            max_epsilon=1.0,
            eps_decay_rate=0.0005, # rate at which exploration drops off
            learning_rate=0.5,
            gamma=0.95,            # discount rate (discount past rewards)
            ep_callback: t.Optional[t.Callable[[int, float], None]]=None
            ):
        """
            Parameters:
            - ep_callback: func(episode_num, current_epsilon) -> None. Called
              at the end of every episode.
        """
        for episode in range(n_training_episodes):
            epsilon = min_epsilon + (
                (max_epsilon - min_epsilon) *
                np.exp(-eps_decay_rate * episode))
            env.reset()
            state = env.copy()
            done = False

            while not done:
                if env.current_player == 'X':
                    action = egreedy_policy(env, self._q_table, epsilon)
                else:
                    action = random.choice(list(env.valid_actions()))
                _, reward, terminated, truncated, _ = env.step(action)
                next_state = env
                done = terminated or truncated

                q = self._q_table.value(envstate(state))
                q_next = 0 if done else self._q_table.value(envstate(next_state))
                q = q + learning_rate * (reward + gamma * q_next - q)
                self._q_table.set_value(envstate(state), q)

                state = next_state.copy()

                if done:
                    self._q_table.set_value(envstate(state), reward)

            if ep_callback:
                ep_callback(episode, epsilon)


def greedy_policy(env: t3.Env, qtable: Qtable):
    """ Greedily select the action that results in the highest value next state """
    best_value = -999999999.9
    best_action = None
    for a in env.valid_actions():
        temp_env = env.copy()
        temp_env.step(a)
        value = qtable.value(envstate(temp_env))
        if value > best_value:
            best_value = value
            best_action = a
    return best_action


def egreedy_policy(env: t3.Env, qtable: Qtable, epsilon: float):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(env, qtable)
    else:
        action = random.choice(list(env.valid_actions()))
    return action


def envstate(env: t3.Env):
    c = ''.join('x' if i == t3.X else 'o' if i == t3.O else '.' for i in env.board)
    return f'{c[:3]}|{c[3:6]}|{c[6:]}'
