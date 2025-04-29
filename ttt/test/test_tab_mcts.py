from pathlib import Path

import pytest
from ttt.agents.mcts import _mcts_decision, random_rollout_reward
from ttt.agents.tab_mcts import TabMctsAgent
import ttt.env as t3


TRAINED_MODELS_DIR = Path('trained_models')


@pytest.mark.skip(reason="fix regular mcts first")
def test_asdf():
    agent = TabMctsAgent.load(TRAINED_MODELS_DIR/'tabmcts_100k_10', 30)
    env = t3.FastEnv.from_str('x.o.xx..o')
    # rr_decision = _mcts_decision(env, 30, random_rollout_reward, False)
    action = agent.get_action(env)
    assert action == 3
