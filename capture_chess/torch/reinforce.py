import time

from matplotlib import pyplot as plt
import numpy as np
import torch
from lib.evaluator import evaluate
from lib.reinforce import PolicyGradientTrainer


def main():
    device = "cpu"
    # train_dbg(device)
    net = train(100, device)
    net.print_summary(device)
    evaluate("policy gradient", net, 10, device)


def train(n_episodes, device):
    print("training...")
    start = time.time()
    net, losses, rewards = PolicyGradientTrainer.train_new_net(n_episodes, device)
    print(f"trained for {time.time() - start:0.1f}s")

    ep_x = list(range(len(losses)))
    plt.title("REINFORCE training")
    plt.xlabel("episodes")
    plt.plot(ep_x, losses, label="avg ep loss")
    plt.plot(ep_x, rewards, label="sum ep reward")
    plt.legend()
    plt.show()

    return net


def train_dbg(device):
    state1 = np.zeros((8,8,8))
    state1[0,0,0] = 1

    state2 = np.zeros((8,8,8))
    state2[7,7,7] = 1

    states = [state1, state2]
    actions = [(0,8), (55,63)]
    rewards = [9.0, 0.0]

    legals1 = np.zeros((4096,))
    legals1[100] = 1
    legals1[200] = 1
    legals1[300] = 1

    legals2 = np.zeros((4096,))
    legals2[1000] = 1
    legals2[2000] = 1
    legals2[3000] = 1

    legal_moves = [legals1, legals2]
    n_steps = 1

    model = PolicyGradientTrainer.train_known_sample(
        states, actions, rewards, legal_moves, n_steps, device
    )

    s1 = torch.from_numpy(state1)
    l1 = torch.from_numpy(legals1)
    probs = model(s1, l1)
    print(probs)


if __name__ == "__main__":
    main()
