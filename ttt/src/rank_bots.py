"""Intended to replace bot showdown.
Train bots elsewhere - this should just be a tournament.
"""

import random
# disable unused import check so u can temporarily comment out bots
# ruff: noqa: F401

from pathlib import Path

from agents.alphazero import AlphaZeroAgent
from agents.mcts import MctsAgent
from agents.perfect import PerfectAgent
from agents.random import RandomAgent
from agents.tab_mcts import TabMctsAgent
from algs.az_evaluators import PvFn
from utils import maths
from utils.maths import is_prob_dist
from utils.ranker import AgentRanker
import algs.az_evaluators as az_eval

PROJECT_ROOT = Path(__file__).parent.parent
TRAINED_MODELS_PATH = PROJECT_ROOT / "trained_models"
N_GAMES = 200


def main():
    agents = [
        # ("Random", RandomAgent()),
        # ("Perfect", PerfectAgent()),
        ("mctsrr10", MctsAgent(n_sims=10)),
        (
            "az10",
            AlphaZeroAgent.load(
                TRAINED_MODELS_PATH / "aznet", n_mcts_sims=10, device="cpu"
            ),
        ),
        # (
        #     "tmcts_sym_100k_30",
        #     TabMctsAgent.load(TRAINED_MODELS_PATH / "tmcts_sym_100k_30", n_sims=30),
        # ),
        # (
        #     "AzTabV30_c1",
        #     AlphaZeroAgent.from_eval(
        #         az_eval.make_uniform_tab_v_eval(), n_mcts_sims=30, c_puct=1.0
        #     ),
        # ),
        # very strong, just a tiny bit worse than cpuct=1
        # ("AzTabV30_c01", AlphaZeroAgent.from_eval(az_eval.make_uniform_tab_v_eval(), n_mcts_sims=30, c_puct=0.1)),
        # ("AzTabPv30_c1", AlphaZeroAgent.from_eval(
        #         az_eval.make_uniform_tab_v_eval(), n_mcts_sims=30, c_puct=1.0
        # )),
        # ("AzTab upv", AlphaZeroAgent.from_eval(
        #     az_eval.make_uniform_tab_v_eval(), n_mcts_sims=30, c_puct=1.0
        # )),
        # (
        #     "AzTab gpv",
        #     AlphaZeroAgent.from_eval(
        #         az_eval.make_greedy_tab_pv_eval(), n_mcts_sims=16, c_puct=1.0
        #     ),
        # ),
        # (
        #     "AzTab noisy gpv",
        #     AlphaZeroAgent.from_eval(
        #         az_eval.make_noisy_pv(az_eval.make_greedy_tab_pv_eval(), 0.8, 0.5),
        #         n_mcts_sims=10,
        #         c_puct=1.0,
        #     ),
        # ),
        # as expected, these play poorly - don't waste time running them
        # ("AzTabV10", make_az_greedy_tab_v_agent(10, c_puct=1.0)),
        # (
        #     "AzUnif10",
        #     AlphaZeroAgent.from_eval(mcts_eval=az_eval.uniform, n_mcts_sims=10),
        # ),
        # (
        #     "AzUnif30",
        #     AlphaZeroAgent.from_eval(mcts_eval=az_eval.uniform, n_mcts_sims=30),
        # ),
        # ("AzRandom10", AlphaZeroAgent.from_eval(mcts_eval=az_eval.random_eval, n_mcts_sims=10)),
        # ("AzRandom30", AlphaZeroAgent.from_eval(mcts_eval=az_eval.random_eval, n_mcts_sims=30)),
    ]

    ranker = AgentRanker(agents)
    stats = ranker.full_round_robin(games_per_matchup=N_GAMES)
    ranker.print_rankings(stats)


if __name__ == "__main__":
    main()
