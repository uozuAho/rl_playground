""" Bot fight!

todo
- add all bots
- support short/long training (short for testing)
- support loading saved models
"""

import os
from pathlib import Path
from ttt.agents.compare import play_and_report
from ttt.agents.mcts import MctsAgent
from ttt.agents.perfect import PerfectAgent
from ttt.agents.random import RandomAgent
from ttt.agents.sb3_dqn import Sb3DqnAgent
from ttt.agents.tab_greedy_v import GreedyVAgent
from ttt.env import TicTacToeEnv


TRAINED_MODEL_DIR = 'trained_models'
Path(TRAINED_MODEL_DIR).mkdir(exist_ok=True)
DO_TRAINING = True
LOAD_SAVED = True


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
    sb3dqn = 'sb3dqn-rng-100'
    sb3dqn_path = Path(TRAINED_MODEL_DIR, f'{sb3dqn}.zip')
    if LOAD_SAVED and os.path.exists(sb3dqn_path):
        agent = Sb3DqnAgent.from_name(sb3dqn)
    else:
        agent = Sb3DqnAgent.train_new(opponent=RandomAgent(), steps=100, save_as=sb3dqn_path)
    agents.append((agent, sb3dqn))

    gv = 'tab-greedy-v-rng'
    gv_path = Path(TRAINED_MODEL_DIR, f'{gv}.json')
    if LOAD_SAVED and os.path.exists(gv_path):
        agent = GreedyVAgent.load(gv_path)
    else:
        agent = GreedyVAgent()
        agent.train(TicTacToeEnv(), 100)
        agent.save(gv_path)
    agents.append((agent, gv))


for a1, l1 in agents:
    for a2, l2 in agents:
        if a1 == a2: continue
        msg = play_and_report(a1, l1, a2, l2, 100, quiet=True)
        print(msg)
        with open("bot-showdown.txt", "a") as f:
            f.write(msg + "\n")
