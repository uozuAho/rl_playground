import time
import torch
from ttt.agents.agent import TttAgent
from ttt.agents.compare import play_and_report
from ttt.agents.random import RandomAgent2
from ttt.agents.torch_greedy_v import GreedyTdAgent
from utils import epsilon


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = 'cpu'
print(f'using device {device}')

agent = GreedyTdAgent(device)
n_train_eps = 10000
start = time.time()
prev = time.time()


def eval_agent(agent: GreedyTdAgent, opponent: TttAgent):
    play_and_report(agent, "greedyTd", opponent, "rando?", 100)


def eval_callback(ep_num):
    global start
    global prev
    if ep_num % 1000 == 0:
        print(f'{time.time() - start:3.1f}s ({time.time() - prev:.2f}s)')
        eval_agent(agent, RandomAgent2())
        prev = time.time()


agent.train(
    opponent=RandomAgent2(),
    n_episodes=n_train_eps,
    buffer_size=32,
    batch_size=16,
    n_ep_update_interval=4,
    epsilon=epsilon.exp_decay_gen(0.5, 0.001, n_train_eps),
    callback=eval_callback
)

path = "conv-greedy-v.pth"
agent.save(path)

agent2 = GreedyTdAgent.load(path, device)
eval_agent(agent2, RandomAgent2())
