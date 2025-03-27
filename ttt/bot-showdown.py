""" Bot fight! """

from ttt.agents.compare import play_and_report
from ttt.agents.mcts import MctsAgent
from ttt.agents.perfect import PerfectAgent
from ttt.agents.random import RandomAgent
from ttt.agents.sb3_dqn import Sb3DqnAgent


DO_TRAINING = True


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


if DO_TRAINING:
    agents.append((Sb3DqnAgent.train_new(opponent=RandomAgent(), steps=100000), 'sb3dqn-rng-100'))


for a1, l1 in agents:
    for a2, l2 in agents:
        if a1 == a2: continue
        msg = play_and_report(a1, l1, a2, l2, 100, quiet=True)
        print(msg)
        with open("bot-showdown.txt", "a") as f:
            f.write(msg + "\n")
