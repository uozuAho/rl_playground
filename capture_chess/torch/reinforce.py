import time

from matplotlib import pyplot as plt
from lib.evaluator import evaluate
from lib.reinforce import PolicyGradientTrainer


def main():
    device = "cpu"
    net = train(1000, device)
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


if __name__ == "__main__":
    main()
