import time
import matplotlib.pyplot as plt

from lib import torch_utils
from lib.evaluator import play_games
from lib.nets import ConvQNet
from lib.trainer import train


def main():
    device = torch_utils.find_device()
    # device = 'cpu'
    print(f"Using device: {device}")
    policy_net = ConvQNet().to(device)
    target_net = ConvQNet().to(device)
    policy_net.print_summary()
    params = {
        "n_episodes": 1000,
        "batch_size": 64,
        "target_net_update_eps": 10,
        "target_net_update_tau": 1.0,
    }
    print("Training with params:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print()
    input("Press enter to continue, ctrl+c to stop training...")
    start = time.time()
    losses, rewards = train(policy_net, target_net, device=device, **params)
    print(f"trained for {time.time()-start}")

    x = list(range(len(losses)))
    plt.xlabel("episodes")
    plt.plot(x, losses, label="loss")
    plt.plot(x, rewards, label="reward")
    plt.legend()
    plt.show()

    avg_reward = play_games(policy_net, 100, device)
    print(f'avg. reward over 100 games: {avg_reward}')


if __name__ == "__main__":
    main()
