
from lib import env
from lib.agent import RandomAgent
from lib.evaluate import evaluate
from lib.greedy_agent import GreedyChessAgent
from lib.mcts import MctsAgent, random_rollout_reward


def main():
    # mcts_vs_random()
    greedy_vs_random()


def mcts_vs_random():
    w = MctsAgent(env.WHITE, n_sims=5, valfn=lambda e,p: random_rollout_reward(e,p,max_depth=5))
    b = RandomAgent(env.BLACK)
    evaluate(w, b, 5)


def greedy_vs_random():
    greedy = GreedyChessAgent(env.WHITE)
    random = RandomAgent(env.BLACK)
    evaluate(greedy, random, 5, halfmove_limit=80)
    print("training greedy agent for 10 episodes...")
    greedy.train_against(random, n_episodes=10, plot=True, halfmove_limit=80)
    evaluate(greedy, random, 5, halfmove_limit=80)


if __name__ == "__main__":
    main()
