""" Bot fight! """

from pathlib import Path
import time
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
from ttt.agents.tab_mcts import TabMctsAgent
from ttt.agents.torch_nn_greedy_v import NnGreedyVAgent
from utils.torch_device import find_device


# TRAIN_FAST = True   # do short training just to verify training works
TRAIN_FAST = False  # do full training to make competent agents
FORCE_TRAIN = False
# FORCE_TRAIN = True  # train agents even if they have a saved model
TRAINED_MODELS_PATH = Path("trained_models")
TRAINED_MODELS_PATH.mkdir(exist_ok=True)
VERBOSE = False
DEVICE = find_device()  # todo: let agents decide what device they use


def main():
    agents = [
        # (RandomAgent(), "random"),
        # (MctsAgent(n_sims=1), "mctsrr1"),
        # (MctsAgent(n_sims=5), "mctsrr5"),
        # (MctsAgent(n_sims=10), "mctsrr10"),
        (MctsAgent(n_sims=30), "mctsrr30"),
        # (PerfectAgent(), "perfect"),
    ]

    # load_or_train_agent(agents, 'tabsarsa-rng', TabSarsaAgent,
    #     lambda: TabSarsaAgent.train_new(RandomAgent(), 100 if TRAIN_FAST else 5000))
    # load_or_train_agent(agents, 'tabqlearn-rng', TabQlearnAgent,
    #     lambda: TabQlearnAgent.train_new(RandomAgent(), 100 if TRAIN_FAST else 5000))
    # load_or_train_agent(agents, 'tabgreedyv-rng', TabGreedyVAgent,
    #     lambda: TabGreedyVAgent.train_new(100 if TRAIN_FAST else 5000))
    # load_or_train_agent(agents, 'tabmcts_20k_20', TabMctsAgent,
    #     train_fn=lambda: TabMctsAgent.train_new(100 if TRAIN_FAST else 20000, n_sims=20),
    #     load_fn=lambda: TabMctsAgent.load(TRAINED_MODELS_PATH/'tabmcts_20k_20', n_sims=20))
    # load_or_train_agent(agents, 'tabmcts_50k_5', TabMctsAgent,
    #     train_fn=lambda: TabMctsAgent.train_new(100 if TRAIN_FAST else 50000, n_sims=5),
    #     load_fn=lambda: TabMctsAgent.load(TRAINED_MODELS_PATH/'tabmcts_50k_5', n_sims=5))
    load_or_train_agent(agents, 'tabmcts_100k_30', TabMctsAgent,
        train_fn=lambda: TabMctsAgent.train_new(100 if TRAIN_FAST else 100000, n_sims=10),
        load_fn=lambda: TabMctsAgent.load(TRAINED_MODELS_PATH/'tabmcts_100k_10', n_sims=30))
    # load_or_train_agent(agents, 'sb3dqn-rng', Sb3DqnAgent,
    #     lambda: Sb3DqnAgent.train_new(opponent=RandomAgent(), steps=100 if TRAIN_FAST else 50000, verbose=VERBOSE))
    # load_or_train_agent(agents, 'sb3ppo-rng', Sb3PpoAgent,
    #     lambda: Sb3PpoAgent.train_new(opponent=RandomAgent(), steps=100 if TRAIN_FAST else 100000, verbose=VERBOSE))
    # load_or_train_agent(agents, 'sb3maskppo-rng', Sb3MaskPpoAgent,
    #     lambda: Sb3MaskPpoAgent.train_new(opponent=RandomAgent(), steps=100 if TRAIN_FAST else 50000, verbose=VERBOSE))
    # load_or_train_agent(agents, 'nngreedyv-rng', NnGreedyVAgent,
    #     lambda: NnGreedyVAgent.train_new(RandomAgent(), 100 if TRAIN_FAST else 5000, DEVICE), DEVICE)
    # load_or_train_agent(agents, 'nngreedymcts10', None, None, DEVICE,
    #     lambda: load_NnGreedyVMctsAgent('nngreedyv-rng', 10))
    # load_or_train_agent(agents, 'nngreedymcts10', None, None, DEVICE,
    #     lambda: load_NnGreedyVMctsAgent('nngreedyv-rng', 20))

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
        train_fn: t.Callable[[], TttAgent] | None,
        device: str | None = None,
        load_fn: t.Callable[[], t.Any] | None = None
        ):
    path = TRAINED_MODELS_PATH/name
    if FORCE_TRAIN and train_fn:
        start = time.time()
        print(f'training {name}...')
        agent = train_fn()
        print(f'done in {time.time() - start:0.1f}s')
        agent.save(path)
        agents.append((agent, name))
    else:
        if load_fn:
            agent = load_fn()
        else:
            agent = try_load(agent_class, path, device)
        if agent:
            agents.append((agent, name))
            return agent


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
    agent = try_load(NnGreedyVAgent, path, DEVICE)
    if agent:
        return MctsAgent(n_sims, lambda e,p: agent.board_val(e.board) * p)


if __name__ == "__main__":
    main()
