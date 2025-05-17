from lib import env
from lib.agent import ChessAgent


def play_game(agent_w: ChessAgent, agent_b: ChessAgent):
    agents = {
        env.WHITE: agent_w,
        env.BLACK: agent_b
    }
    game = env.ChessGame()
    done = False
    total_white_reward = 0.0
    while not done:
        agent = agents[game.turn]
        move = agent.get_action(game)
        done, white_reward = game.step(move)
        total_white_reward += white_reward
    return total_white_reward


def evaluate(agent_w: ChessAgent, agent_b: ChessAgent, n_games=10):
    print(f"evaulating over {n_games} games...")
    white_rewards = []
    for _ in range(n_games):
        white_reward = play_game(agent_w, agent_b)
        white_rewards.append(white_reward)
    print(f"avg white reward {sum(white_rewards)/len(white_rewards)}")
