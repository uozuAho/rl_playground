from ttt.agents.compare import play_and_report
from ttt.agents.mcts import MctsAgent
from ttt.agents.perfect import PerfectAgent
from ttt.agents.random import RandomAgent


play_and_report(RandomAgent(), "rando", RandomAgent(), "rando", 1000)       # x wins 60%
play_and_report(MctsAgent(n_sims=1), "mcts1", RandomAgent(), "rando", 50)   # tiny bit better
play_and_report(MctsAgent(n_sims=5), "mcts5", RandomAgent(), "rando", 50)   # ~88%
play_and_report(MctsAgent(n_sims=10), "mcts10", RandomAgent(), "rando", 50) # ~94%
play_and_report(RandomAgent(), "rando", MctsAgent(n_sims=10), "mcts10", 50)   # o wins ~68%
play_and_report(RandomAgent(), "rando", MctsAgent(n_sims=50), "mcts50", 50)   # o wins ~90%
play_and_report(MctsAgent(n_sims=10), "mcts10", PerfectAgent(), "perfect", 50)  # o wins ~50%
play_and_report(MctsAgent(n_sims=100), "mcts100", PerfectAgent(), "perfect", 50) # o wins ~34%
play_and_report(MctsAgent(n_sims=200), "mcts200", PerfectAgent(), "perfect", 50) # Slow! o wins ~32%
