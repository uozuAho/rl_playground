
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
    halfmove_limit = 80
    capture_reward_factor = 0.001
    n_train_episodes = 20
    n_eval_episodes = 5

    greedy = GreedyChessAgent(env.WHITE)
    random = RandomAgent(env.BLACK)
    evaluate(greedy, random, n_eval_episodes, halfmove_limit=halfmove_limit)
    print(f"training greedy agent for {n_train_episodes} episodes...")
    greedy.train_against(
        random,
        n_episodes=n_train_episodes,
        capture_reward_factor=capture_reward_factor,
        halfmove_limit=halfmove_limit,
        plot=True)
    evaluate(greedy, random, n_eval_episodes, halfmove_limit=halfmove_limit)


if __name__ == "__main__":
    main()
