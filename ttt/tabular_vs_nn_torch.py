""" Trying to figure out why my NN greedy-v agent doesn't learn, while my
tabular one does
"""


import torch
from ttt.agents.compare import play_and_report
from ttt.agents.nn_greedy_v import NnGreedyVAgent
from ttt.agents.random import RandomAgent
import ttt.env
from ttt.agents.tab_greedy_v import GreedyVAgent
from ttt.env import TicTacToeEnv


tab_agent = GreedyVAgent(allow_invalid_actions=False)
env = TicTacToeEnv(
    on_invalid_action=(ttt.env.INVALID_ACTION_GAME_OVER
                       if tab_agent.allow_invalid_actions
                       else ttt.env.INVALID_ACTION_THROW)
)

def eval_callback(agent, ep_num, epsilon):
    if ep_num > 10 and ep_num % 1000 == 0:
        play_and_report(agent, "tabular greedy v", RandomAgent(), "random", 100)

tab_agent.train(
    env,
    n_training_episodes=3000,
    eps_decay_rate=0.001,
    learning_rate=0.1,
    ep_callback=lambda ep, eps: eval_callback(tab_agent, ep, eps))

tab_agent.save('tabular-greedy-v.json')



device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = 'cpu'
print(f'using device {device}')
nn_agent = NnGreedyVAgent(device)
nn_agent.train(
    env,
    n_training_episodes=3000,
    min_epsilon=0.05,
    max_epsilon=1.0,
    ep_callback=lambda ep, eps: eval_callback(nn_agent, ep, eps))

env.reset()
print(tab_agent.action_values(env))
