
from lib import env
from lib.agent import RandomAgent
from lib.evaluate import evaluate
from lib.mcts import MctsAgent, random_rollout_reward


def main():
    w = MctsAgent(env.WHITE, n_sims=5, valfn=lambda e,p: random_rollout_reward(e,p,max_depth=5))
    b = RandomAgent(env.BLACK)
    evaluate(w, b, 1)


if __name__ == "__main__":
    main()
