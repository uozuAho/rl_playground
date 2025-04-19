from ttt.agents.mcts import MctsAgent
import ttt.env as t3


def test_plays_a_game():
    env = t3.Env()
    agent = MctsAgent(n_sims=1)
    while env.status() == t3.IN_PROGRESS:
        action = agent.get_action(env)
        env.step(action)
