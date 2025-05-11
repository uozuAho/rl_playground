from dataclasses import dataclass
from pathlib import Path
import time
import typing as t
import matplotlib.pyplot as plt

from lib import torch_utils
from lib.evaluator import evaluate
from lib.nets import ChessNet, ConvQNet, LinearFCQNet
import lib.trainer


TRAINED_MODEL_DIR = Path("./trained_models")
FORCE_TRAIN = True
# DEVICE = torch_utils.find_device()
DEVICE = 'cpu'


# path, device -> None (loads agent nets to agent object)
type LoadFn = t.Callable[[str, str], ChessNet]


@dataclass
class Agent:
    label: str
    policy_net: ChessNet
    target_net: ChessNet
    device: str
    train_params: dict
    load_fn: LoadFn | None = None

    @property
    def trained_model_path(self):
        return TRAINED_MODEL_DIR / self.label

    @property
    def can_load(self):
        return self.trained_model_path.exists() and self.load_fn


AGENTS = [
    # q-learning with FC NN, trains slowly, doesn't do well
    # Agent(
    #     label="LinearFC",
    #     device=DEVICE,
    #     policy_net=LinearFCQNet().to(DEVICE),
    #     target_net=LinearFCQNet().to(DEVICE),
    #     train_params={
    #         "n_episodes": 1000,
    #     },
    # ),
    # q-learning with convolutional net. Does better than linear.
    # Trains from 1000 episodes in ~90 seconds.
    Agent(
        label="ConvQNet",
        device=DEVICE,
        policy_net=ConvQNet().to(DEVICE),
        target_net=ConvQNet().to(DEVICE),
        train_params={
            "n_episodes": 1000,
            "batch_size": 64,
            "target_net_update_eps": 10,
            "target_net_update_tau": 1.0,
        },
        load_fn=ConvQNet.load,
    ),
]


def main():
    for agent in AGENTS:
        if agent.can_load and not FORCE_TRAIN:
            print(f"loading {agent.label}")
            agent.policy_net = agent.load_fn(agent.trained_model_path, agent.device)
        else:
            assert agent.policy_net
            assert agent.target_net

            agent.policy_net.print_summary(agent.device)

            print(f"Training {agent.label} with params:")
            for k, v in agent.train_params.items():
                print(f"  {k}: {v}")
            print()
            input("Press enter to continue. Press ctrl+c to stop training early")

            train(agent)

        try:
            evaluate(agent.label, agent.policy_net, 100, agent.device)
        except Exception as e:
            print(f"{agent.label} failed evaluation: {e}")

        # todo: ctrl+c only stops after hitting another key
        input("Press enter to continue to next agent, ctrl+c to exit")


def train(agent: Agent):
    eval_period_ep = 50
    eval_rewards = []

    def interim_eval(ep_num: int):
        if ep_num % eval_period_ep == 0:
            try:
                avg_reward = evaluate(agent.label, agent.policy_net, 5, agent.device)
                eval_rewards.append(avg_reward)
            except Exception as e:
                print(f"nonfatal: eval error: {e}")
                eval_rewards.append(0)

    start = time.time()
    losses, rewards = lib.trainer.train(
        agent.policy_net,
        agent.target_net,
        device=agent.device,
        ep_callback=interim_eval,
        **agent.train_params,
    )
    agent.policy_net.save(agent.trained_model_path)

    print(f"{agent.label} trained for {time.time() - start:0.1f}s")

    ep_x = list(range(len(losses)))
    plt.title(f"{agent.label} training")
    plt.xlabel("episodes")
    plt.plot(ep_x, losses, label="avg ep loss")
    plt.plot(ep_x, rewards, label="sum ep reward")

    ev_x = list(x * eval_period_ep for x in range(len(eval_rewards)))
    plt.plot(ev_x, eval_rewards, label="avg eval reward")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
