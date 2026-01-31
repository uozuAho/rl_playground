from agents.mcts_agent import make_uniform_agent
from agents.rando import RandomAgent
from utils.play import play_games_parallel

ma = make_uniform_agent(n_sims=45)
rng = RandomAgent()
w,ll,d = play_games_parallel(ma, rng, 20)
print(w,ll,d)
