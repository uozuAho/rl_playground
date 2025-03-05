from abc import ABC
import random
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.ppo_mask import MaskablePPO

import ccenv


def make_env():
    return ccenv.CaptureChess(illegal_action_behaviour=ccenv.ILLEGAL_ACTION_THROW)


def train(name, steps):
    env = make_vec_env(lambda: make_env(), n_envs=16)

    model = MaskablePPO(
        policy='MlpPolicy',
        batch_size=64,
        gamma=0.99,
        learning_rate=0.01,
        env = env,
        verbose=1
    )
    model.learn(total_timesteps=steps, log_interval=steps//30)
    return model


class Agent(ABC):
    def get_action(self, env: ccenv.CaptureChess):
        raise NotImplementedError()


class RandomAgent(Agent):
    def get_action(self, env: ccenv.CaptureChess):
        return random.choice(list(env.legal_actions()))


class TrainedAgent(Agent):
    def __init__(self, model: MaskablePPO):
        self._model = model

    def get_action(self, env):
        nn_out, _ = model.predict(env.current_obs(), deterministic=True)
        return nn_out.argmax()


def play_game(agent: Agent, env: ccenv.CaptureChess, interactive=False):
    done = False
    total_reward = 0
    turn = 0
    while not done:
        turn += 1
        action = agent.get_action(env)
        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        total_reward += reward
        if interactive:
            env.render()
            input(f"turn {turn}. press a key...")
    return total_reward


try:
    model = train("maskppo", 100)
    agent = TrainedAgent(model)
    # play_game(agent, make_env(), interactive=True)
except ccenv.IllegalActionError as e:
    print("Illegal move attempted")
    print(e.move)
    print(e.board.board)
    print(e.board.board.fen())
