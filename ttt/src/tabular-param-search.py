"""Not as smart as optuna - just try all combinations of given params"""

from collections import defaultdict
import itertools
from agents.qlearn import TabQlearnAgent
from agents.sarsa import TabSarsaAgent
from agents.random import RandomAgent
import matplotlib.pyplot as plt
import ttt.env


def make_env():
    return ttt.env.EnvWithOpponent(
        opponent=RandomAgent(), invalid_action_response=ttt.env.INVALID_ACTION_GAME_OVER
    )


learning_rates = [0.05, 0.2]
min_eps = [0.01, 0.1]
max_eps = [1.0]
eps_decays = [0.0005, 0.002]
gammas = [0.95, 1.0]
agents = ["sarsa", "qlearn"]

n_train_eps = 30000
eval_period = 1000

current_agent = None
current_name = ""
data = defaultdict(list)


def my_eval(agent, num_games=50):
    env = make_env()
    total_reward = 0
    for _ in range(num_games):
        env.reset()
        done = False
        while not done:
            action = agent.get_action(env)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total_reward += reward
    return total_reward / num_games


def eval_callback(ep_num, epsilon):
    global data
    if ep_num % eval_period == 0:
        avg_reward = my_eval(current_agent)
        data[current_name].append(avg_reward)
        print(avg_reward)


for lr, mne, mxe, ed, g, a in itertools.product(
    learning_rates, min_eps, max_eps, eps_decays, gammas, agents
):
    current_agent = (
        TabSarsaAgent(allow_invalid_actions=True)
        if a == "sarsa"
        else TabQlearnAgent(allow_invalid_actions=True)
    )
    current_name = a + "," + ",".join(f"{x:0.3f}" for x in [lr, mne, mxe, ed, g])
    currently_training = current_name
    print(f"training {current_name} for {n_train_eps} eps...")
    current_agent.train(
        make_env(),
        RandomAgent(),
        n_training_episodes=n_train_eps,
        min_epsilon=mne,
        max_epsilon=mxe,
        eps_decay_rate=ed,
        gamma=g,
        ep_callback=eval_callback,
    )

# todo: the resulting chart is too busy. write to csv and analyse there?
# idea
# do in parallel for funsies
# plot:
#   show/hide line in plot
#   show sum(all eval rewards) in legend
# output to csv anyway
ep_nums = [x + eval_period for x in range(0, n_train_eps, eval_period)]
for name, vals in data.items():
    plt.plot(ep_nums, vals, label=name)
plt.legend()
plt.title("avg return vs eps")
plt.xlabel("trained eps")
plt.ylabel("avg return")
plt.show()
