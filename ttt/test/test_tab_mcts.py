from pathlib import Path
from ttt.agents.tab_mcts import TabMctsAgent
import ttt.env as t3


TRAINED_MODELS_DIR = Path('trained_models')


def test_asdf():
    agent = TabMctsAgent.load(TRAINED_MODELS_DIR/'tabmcts_100k_10', 30)
    env = t3.FastEnv.from_str('x.o.xx..o')
    action = agent.get_action(env)
    assert action == 3
