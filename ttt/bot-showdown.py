""" Bot fight!

todo
- add all bots
- support short/long training (short for testing)
- support loading saved models
"""

from pathlib import Path
import typing as t

from ttt.agents.agent import TttAgent
from ttt.agents.compare import play_and_report
from ttt.agents.mcts import MctsAgent
from ttt.agents.perfect import PerfectAgent
from ttt.agents.random import RandomAgent, RandomAgent2
from ttt.agents.sarsa import SarsaAgent
from ttt.agents.sb3_dqn import Sb3DqnAgent
from ttt.agents.tab_greedy_v import TabGreedyVAgent
from ttt.agents.torch_nn_greedy_v import NnGreedyVAgent
from utils.torch_device import find_device


TRAINED_MODELS_PATH = Path("trained_models")
TRAINED_MODELS_PATH.mkdir(exist_ok=True)
TRAIN_FAST = True   # do short training just to verify training works
# TRAIN_FAST = False  # do full training to make competent agents
VERBOSE = False
DEVICE = find_device()


def main():
    agents = [
        (RandomAgent(), "random"),
        (MctsAgent(n_sims=1), "mcts1"),
        # (MctsAgent(n_sims=5), "mcts5"),
        # (MctsAgent(n_sims=10), "mcts10"),
        (PerfectAgent(), "perfect"),

        # too slow
        # (MctsAgent(n_sims=50), "mcts50"),
        # (MctsAgent(n_sims=100), "mcts100"),
        # (MctsAgent(n_sims=200), "mcts200"),
    ]

    load_or_train_agent(agents, 'tabsarsa-rng', SarsaAgent,
        lambda: SarsaAgent.train_new(RandomAgent(), 100))
    load_or_train_agent(agents, 'tabgreedyv-rng', TabGreedyVAgent,
        lambda: TabGreedyVAgent.train_new(100))
    load_or_train_agent(agents, 'sb3dqn-rng', Sb3DqnAgent,
        lambda: Sb3DqnAgent.train_new(opponent=RandomAgent(), steps=100, verbose=VERBOSE))
    load_or_train_agent(agents, 'nngreedyv-rng', NnGreedyVAgent,
        lambda: NnGreedyVAgent.train_new(RandomAgent2(), 100, DEVICE), DEVICE)

    for a1, l1 in agents:
        for a2, l2 in agents:
            if a1 == a2: continue
            msg = play_and_report(a1, l1, a2, l2, 100, quiet=True)
            print(msg)
            with open("bot-showdown.txt", "a") as f:
                f.write(msg + "\n")


def load_or_train_agent(
        agents,
        name,
        agent_class,
        train_fn: t.Callable[[], TttAgent],
        device: str | None = None
        ):
    path = TRAINED_MODELS_PATH/name
    agent = try_load(agent_class, path, device)
    if agent:
        agents.append((agent, name))
        return agent
    agent = train_fn()
    agent.save(path)
    agents.append((agent, name))


def try_load(agent_class, path, device=None):
    try:
        if device:
            return agent_class.load(path, device)
        else:
            return agent_class.load(path)
    except FileNotFoundError:
        print(f'{path}: no saved model')
        return None



if __name__ == "__main__":
    main()
