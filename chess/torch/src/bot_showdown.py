from env import env
from agents.agent import RandomAgent
from agents.andoma.andoma_agent import AndomaAgent, AndomaMctsAgent
from utils.evaluate import evaluate
from agents.greedy_agent import GreedyChessAgent
from agents.mcts import MctsAgent, random_rollout_reward


def main():
    # mcts_vs_random()
    # greedy_vs_random()
    # greedy_vs_andoma()
    andoma_vs_andoma_mcts()


def mcts_vs_random():
    w = MctsAgent(
        env.WHITE, n_sims=5, valfn=lambda e, p: random_rollout_reward(e, p, max_depth=5)
    )
    b = RandomAgent(env.BLACK)
    evaluate(w, b, 5)


def greedy_vs_random():
    halfmove_limit = 80
    capture_reward_factor = 0.001
    n_train_episodes = 200
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
        plot=True,
    )
    evaluate(greedy, random, n_eval_episodes, halfmove_limit=halfmove_limit)


# def greedy_vs_andoma():
#     halfmove_limit = 100
#     capture_reward_factor = 0.001
#     n_train_episodes = 2
#     n_eval_episodes = 5

#     greedy = GreedyChessAgent(env.WHITE)
#     andoma = AndomaAgent(env.BLACK, search_depth=2)
#     # evaluate(greedy, andoma, n_eval_episodes, halfmove_limit=halfmove_limit)
#     print(f"training greedy agent for {n_train_episodes} episodes...")
#     greedy.train_against(
#         andoma,
#         n_episodes=n_train_episodes,
#         capture_reward_factor=capture_reward_factor,
#         halfmove_limit=halfmove_limit,
#         plot=False,
#     )
# evaluate(greedy, andoma, n_eval_episodes, halfmove_limit=halfmove_limit)


def andoma_vs_andoma_mcts():
    a = AndomaAgent(env.WHITE, search_depth=2)
    m = AndomaMctsAgent(env.BLACK, n_sims=30)
    print("andoma vs andoma_mcts:")
    evaluate(a, m, 5, halfmove_limit=100)


if __name__ == "__main__":
    main()
