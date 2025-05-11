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
DEVICE = torch_utils.find_device()


# path, device -> ChessNet
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
        return TRAINED_MODEL_DIR/self.label


AGENTS = [
    Agent(
        label="LinearFC",
        device=DEVICE,
        policy_net=LinearFCQNet().to(DEVICE),
        target_net=LinearFCQNet().to(DEVICE),
        train_params={
            "n_episodes": 1000,
        },
    ),
    Agent(
        label="ConvQNet",
        device=DEVICE,
        policy_net=ConvQNet().to(DEVICE),
        target_net=ConvQNet().to(DEVICE),
        train_params= {
            "n_episodes": 1000,
            "batch_size": 64,
            "target_net_update_eps": 10,
            "target_net_update_tau": 1.0,
        },
        load_fn=ConvQNet.load
    )
]


def main():
    for agent in AGENTS:
        agent.policy_net.print_summary()

        print(f"Training {agent.label} with params:")
        for k, v in agent.train_params.items():
            print(f"  {k}: {v}")
        print()
        input("Press enter to continue, ctrl+c to stop training...")

        load_or_train(agent)

        try:
            evaluate(agent.label, agent.policy_net, 100, agent.device)
        except Exception as e:
            print(f"{agent.label} failed evaluation: {e}")

        input("Press enter to continue to next agent, ctrl+c to exit")


def load_or_train(agent: Agent):
    if not FORCE_TRAIN and agent.trained_model_path.exists() and agent.load_fn:
        print(f"loading {agent.label}")
        return agent.load_fn(agent.trained_model_path, agent.device)
    return train(agent)


def train(agent: Agent):
    def interim_eval(ep_num: int):
        if ep_num > 64 and ep_num % 50 == 0:
            try:
                evaluate(agent.label, agent.policy_net, 5, agent.device)
            except Exception as e:
                print(f"nonfatal: eval error: {e}")

    start = time.time()
    losses, rewards = lib.trainer.train(
        agent.policy_net, agent.target_net, device=agent.device, ep_callback=interim_eval, **agent.train_params
    )
    agent.policy_net.save(agent.trained_model_path)

    print(f"{agent.label} trained for {time.time() - start:0.1f}s")

    x = list(range(len(losses)))
    plt.title(f'{agent.label} training')
    plt.xlabel("episodes")
    plt.plot(x, losses, label="loss")
    plt.plot(x, rewards, label="reward")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
