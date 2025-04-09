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
from ttt.agents.qlearn import TabQlearnAgent
from ttt.agents.random import RandomAgent
from ttt.agents.sarsa import TabSarsaAgent
from ttt.agents.sb3_dqn import Sb3DqnAgent
from ttt.agents.sb3_maskppo import Sb3MaskPpoAgent
from ttt.agents.sb3ppo import Sb3PpoAgent
from ttt.agents.tab_greedy_v import TabGreedyVAgent
from ttt.agents.torch_nn_greedy_v import NnGreedyVAgent
from ttt.agents.torch_nn_greedy_v_mcts import NnGreedyVMctsAgent
from utils.torch_device import find_device


TRAIN_FAST = True   # do short training just to verify training works
# TRAIN_FAST = False  # do full training to make competent agents
TRAINED_MODELS_PATH = Path("trained_models")
TRAINED_MODELS_PATH.mkdir(exist_ok=True)
VERBOSE = False
DEVICE = find_device()  # todo: let agents decide what device they use


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

    load_or_train_agent(agents, 'tabsarsa-rng', TabSarsaAgent,
        lambda: TabSarsaAgent.train_new(RandomAgent(), 100))
    load_or_train_agent(agents, 'tabqlearn-rng', TabQlearnAgent,
        lambda: TabQlearnAgent.train_new(RandomAgent(), 100))
    load_or_train_agent(agents, 'tabgreedyv-rng', TabGreedyVAgent,
        lambda: TabGreedyVAgent.train_new(100))
    load_or_train_agent(agents, 'sb3dqn-rng', Sb3DqnAgent,
        lambda: Sb3DqnAgent.train_new(opponent=RandomAgent(), steps=100, verbose=VERBOSE))
    load_or_train_agent(agents, 'sb3ppo-rng', Sb3PpoAgent,
        lambda: Sb3PpoAgent.train_new(opponent=RandomAgent(), steps=100, verbose=VERBOSE))
    load_or_train_agent(agents, 'sb3maskppo-rng', Sb3MaskPpoAgent,
        lambda: Sb3MaskPpoAgent.train_new(opponent=RandomAgent(), steps=100, verbose=VERBOSE))
    load_or_train_agent(agents, 'nngreedyv-rng', NnGreedyVAgent,
        lambda: NnGreedyVAgent.train_new(RandomAgent(), 100, DEVICE), DEVICE)
    load_or_train_agent(agents, 'nngreedymcts1', None, None, DEVICE,
        lambda: load_NnGreedyVMctsAgent('nngreedyv-rng', 1))
    load_or_train_agent(agents, 'nngreedymcts100', None, None, DEVICE,
        lambda: load_NnGreedyVMctsAgent('nngreedyv-rng', 100))

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
        device: str | None = None,
        load_fn: t.Callable[[], t.Any] | None = None
        ):
    if load_fn:
        agent = load_fn()
    else:
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


def load_NnGreedyVMctsAgent(name, n_sims):
    path = TRAINED_MODELS_PATH/name
    agent = try_load(NnGreedyVMctsAgent, path, DEVICE)
    if agent:
        agent.n_simulations = n_sims
        return agent


if __name__ == "__main__":
    main()
