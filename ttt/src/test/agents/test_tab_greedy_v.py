from agents.compare import play_game
from agents.random import RandomAgent
from agents.tab_greedy_v import TabGreedyVAgent


def test_train_load_play():
    a = TabGreedyVAgent.train_new(n_eps=100)
    save_path = "/tmp/tab_greedy_v"
    a.save(save_path)
    a2 = TabGreedyVAgent.load(save_path)
    play_game(a2, RandomAgent())
