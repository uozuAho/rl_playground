import time
from ttt.agents.agent import TttAgent2
from ttt.agents.compare import play_and_report2
from ttt.agents.random import RandomAgent2
from ttt.agents.torch_greedy_v import GreedyTdAgent
from utils import epsilon
from utils.torch_device import find_device


# N_TRAIN_EPS = 10000
N_TRAIN_EPS = 100  # for quick testing


device = find_device()
agent = GreedyTdAgent(device)
start = time.time()
prev = time.time()


def eval_agent(agent: GreedyTdAgent, opponent: TttAgent2):
    play_and_report2(agent, "greedyTd", opponent, "rando?", 100)


def eval_callback(ep_num):
    global start
    global prev
    if ep_num % 1000 == 0:
        print(f'{time.time() - start:3.1f}s ({time.time() - prev:.2f}s)')
        eval_agent(agent, RandomAgent2())
        prev = time.time()


agent.train(
    opponent=RandomAgent2(),
    n_episodes=N_TRAIN_EPS,
    buffer_size=32,
    batch_size=16,
    n_ep_update_interval=4,
    epsilon=epsilon.exp_decay_gen(0.5, 0.001, N_TRAIN_EPS),
    callback=eval_callback
)

path = "conv-greedy-v.pth"
agent.save(path)
agent2 = GreedyTdAgent.load(path, device)
eval_agent(agent2, RandomAgent2())
