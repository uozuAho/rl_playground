import random
import typing as t

from agents.agent import TttAgent
from agents.mcts import mcts_decision
from agents.tab_greedy_v import StateValueTable
import ttt.env as t3
from utils import epsilon as epsfuncs


type EpCallback = t.Callable[[int], None]  # (episode number) -> None


class TabMctsAgent(TttAgent):
    """
    Tabular MCTS learning agent. Uses a state-value table as the
    rollout/simulation value step in MCTS.
    """

    def __init__(self, n_sims: int, q_table: StateValueTable | None = None):
        self._q_table = q_table or StateValueTable()
        self.n_sims = n_sims

    @staticmethod
    def load(path, n_sims: int):
        agent = TabMctsAgent(n_sims, StateValueTable.load(path))
        return agent

    @staticmethod
    def train_new(n_eps: int, n_sims: int, use_symmetries=False):
        agent = TabMctsAgent(n_sims)
        agent.train(t3.TttEnv(), n_eps, n_sims=n_sims, use_symmetries=use_symmetries)
        return agent

    def get_action(self, env: t3.TttEnv):
        return self._mcts_policy(env, self._q_table, self.n_sims)

    def save(self, path):
        self._q_table.save(path)

    def _mcts_policy(self, env: t3.TttEnv, qtable: StateValueTable, n_sims: int):
        return mcts_decision(env, n_sims, self._val_estimate, True)

    def _emcts_policy(
        self, env: t3.TttEnv, qtable: StateValueTable, n_sims: int, epsilon: float
    ):
        random_num = random.uniform(0, 1)
        if random_num > epsilon:
            action = self._mcts_policy(env, qtable, n_sims)
        else:
            action = random.choice(list(env.valid_actions()))
        return action

    def _val_estimate(self, env: t3.TttEnv, player: t3.Player):
        return self._q_table.value(env) * player

    def train(
        self,
        env: t3.TttEnv,
        n_training_episodes,
        eps_start=0.99,
        eps_end=0,
        learning_rate=0.5,
        gamma=0.95,
        n_sims=20,
        use_symmetries=False,  # cheat by learning all symmetrical boards
        ep_callback: t.Optional[EpCallback] = None,
    ):
        eps_gen = epsfuncs.exp_decay_gen(eps_start, eps_end, n_training_episodes)
        for episode in range(n_training_episodes):
            env.reset()
            state = env.copy()
            done = False
            epsilon = eps_gen.__next__()

            while not done:
                action = self._emcts_policy(env, self._q_table, n_sims, epsilon)
                _, reward, terminated, truncated, _ = env.step(action)
                next_state = env
                done = terminated or truncated

                q = self._q_table.value(state)
                q_next = 0 if done else self._q_table.value(next_state)
                q = q + learning_rate * (reward + gamma * q_next - q)
                if use_symmetries:
                    for sym in t3.symmetrics(state):
                        self._q_table.set_value(sym, q)
                else:
                    self._q_table.set_value(state, q)

                if done:
                    self._q_table.set_value(next_state, reward)

                state = next_state.copy()

            if ep_callback:
                ep_callback(episode)
