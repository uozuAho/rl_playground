import json
import random
import typing as t

from ttt.agents.agent import TttAgent
from ttt.agents.mcts import _mcts_decision
from ttt.agents.tab_greedy_v import Qtable
import ttt.env as t3
from utils import epsilon as epsfuncs


type EpCallback = t.Callable[[int], None]   # (episode number) -> None


class TabMctsAgent(TttAgent):
    """
    Tabular MCTS learning agent.
    """
    def __init__(self, n_sims: int, q_table: Qtable | None = None):
        self._q_table = q_table or Qtable()
        self.n_sims = n_sims

    @staticmethod
    def load(path, n_sims: int):
        with open(path) as infile:
            data = json.loads(infile.read())
        agent = TabMctsAgent(n_sims, Qtable(data))
        return agent

    @staticmethod
    def train_new(n_eps: int, n_sims):
        agent = TabMctsAgent(n_sims)
        agent.train(t3.Env(), n_eps, n_sims=n_sims)
        return agent

    def get_action(self, env: t3.Env):
        return mcts_policy(env, self._q_table, self.n_sims)

    def save(self, path):
        self._q_table.save(path)

    def train(self,
            env: t3.Env,
            n_training_episodes,
            eps_start=0.99,
            eps_end=0,
            learning_rate=0.1,
            gamma=0.95,
            n_sims=20,
            ep_callback: t.Optional[EpCallback]=None
            ):
        eps_gen = epsfuncs.exp_decay_gen(eps_start, eps_end, n_training_episodes)
        for episode in range(n_training_episodes):
            env.reset()
            state = env.copy()
            done = False
            epsilon = eps_gen.__next__()

            while not done:
                action = emcts_policy(env, self._q_table, n_sims, epsilon)
                _, reward, terminated, truncated, _ = env.step(action)
                next_state = env
                done = terminated or truncated

                q = self._q_table.value(state.str1d())
                q_next = 0 if done else self._q_table.value(next_state.str1d())
                q = q + learning_rate * (reward + gamma * q_next - q)
                self._q_table.set_value(state.str1d(), q)

                state = next_state.copy()

                if done:
                    self._q_table.set_value(state.str1d(), reward)

            if ep_callback:
                ep_callback(episode)


def mcts_policy(env: t3.Env, qtable: Qtable, n_sims: int):
    return _mcts_decision(env, n_sims, lambda e,p: qtable.value(e.str1d()) * p)


def emcts_policy(env: t3.Env, qtable: Qtable, n_sims: int, epsilon: float):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = mcts_policy(env, qtable, n_sims)
    else:
        action = random.choice(list(env.valid_actions()))
    return action
