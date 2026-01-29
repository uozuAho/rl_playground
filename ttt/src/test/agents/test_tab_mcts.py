from agents.compare import play_game
from agents.random import RandomAgent
from agents.tab_mcts import TabMctsAgent


def test_train_load_play():
    a = TabMctsAgent.train_new(n_eps=100, n_sims=30)
    save_path = "/tmp/tab_mcts"
    a.save(save_path)
    a2 = TabMctsAgent.load(save_path, n_sims=30)
    play_game(a2, RandomAgent())
