"""
Trying to figure out how best to train tabular agents.
"""

import time
from ttt.agents.tab_greedy_v import TabGreedyVAgent
import ttt.env
from ttt.env import TicTacToeEnv
from ttt.agents.random import RandomAgent
from ttt.agents.sarsa import SarsaAgent
from ttt.agents.qlearn import QlearnAgent


def my_eval(a, opponent, num_games=20):
    env = TicTacToeEnv(
        opponent=opponent or RandomAgent(),
        on_invalid_action=ttt.env.INVALID_ACTION_GAME_OVER if a.allow_invalid_actions else ttt.env.INVALID_ACTION_THROW
    )
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0
    for _ in range(num_games):
        env.reset()
        done = False
        reward = 0
        while not done:
            action = a.get_action(env)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total_reward += reward
        wins += 1 if reward == 1 else 0
        losses += 1 if reward == -1 else 0
        draws += 1 if reward == 0 else 0
    avg_reward = total_reward / num_games
    # print(f'{num_games} games. wins: {wins}, draws: {draws}, losses: {losses}. Avg reward: {avg_reward}')
    return avg_reward


def eval_vs_opponents(agents, opponents, num_games):
    for aname, a in agents:
        for oname, o in opponents:
            print(f'{aname} vs {oname}:')
            my_eval(a, o, num_games)


def train_eval(agent: SarsaAgent | QlearnAgent | TabGreedyVAgent, opponent, total_eps, eval_interval):
    ep_nums = []
    epsilons = []
    returns = []
    times = []
    t_last = time.time()
    def callback(ep_num, epsilon):
        nonlocal t_last
        if ep_num % eval_interval == 0:
            avg = my_eval(agent, opponent, num_games=40)
            train_time = time.time() - t_last
            ep_nums.append(ep_num)
            epsilons.append(epsilon)
            returns.append(avg)
            times.append(train_time)
            print(f'{train_time:0.1f}: ep {ep_num}. avg return {avg}')
            # print(f'avg return: {avg}')
            # print(f'table size: {agent._q_table.size()}')
            # print(f'time: {time.time() - t_last}')
            t_last = time.time()
    try:
        env = TicTacToeEnv(
            opponent=opponent,
            on_invalid_action=ttt.env.INVALID_ACTION_GAME_OVER if agent.allow_invalid_actions else ttt.env.INVALID_ACTION_THROW)
        agent.train(env, total_eps, eps_decay_rate=0.001, learning_rate=0.1, ep_callback=callback)
    except KeyboardInterrupt:
        pass
    return ep_nums, returns, epsilons
    # plt.plot(ep_nums, returns, label='avg return')
    # plt.plot(ep_nums, epsilons, label='epsilon')
    # plt.legend()
    # plt.xlabel('episodes')
    # plt.show()


# rng = train_sarsa_agent(SarsaAgent(), RandomAgent(), n_train_eps=1000)
# perf = train_sarsa_agent(SarsaAgent(), PerfectAgent(), n_train_eps=1000)
# rper = train_sarsa_agent(SarsaAgent(), RandomAgent(), n_train_eps=500)
# rper = train_sarsa_agent(rper, PerfectAgent(), n_train_eps=500)
# eval_vs_opponents(
#     [('rngo', rng), ('perfo', perf), ('rng then perf', rper)],
#     [('random', RandomAgent()), ('perfect', PerfectAgent())],
#     num_games=50
# )
# sen, sr, se = train_eval(SarsaAgent(allow_invalid_actions=True), RandomAgent(), total_eps=10000, eval_interval=1000)
# qen, qr, qe = train_eval(QlearnAgent(allow_invalid_actions=True), RandomAgent(), total_eps=10000, eval_interval=1000)
agent = TabGreedyVAgent(allow_invalid_actions=False)
qen, qr, qe = train_eval(agent, opponent=None, total_eps=10000, eval_interval=1000)
# agent.save('asdf.json')
# agent = GreedyVAgent.load('asdf.json')
# my_eval(agent, RandomAgent())
# train_eval(SarsaAgent(), PerfectAgent(), total_eps=10000, eval_interval=1000)
# plt.plot(sen, sr, label='sarsa avg return')
# plt.plot(sen, se, label='sarsa epsilon')
# plt.plot(qen, qr, label='qlearn avg return')
# plt.plot(qen, qe, label='qlearn epsilon')
# plt.legend()
# plt.xlabel('episodes')
# plt.show()
