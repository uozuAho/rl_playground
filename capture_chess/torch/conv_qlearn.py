import torch

from lib.nets import ConvQNet
from lib.trainer import train


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
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
    train(policy_net, target_net, device=device, **params)


if __name__ == "__main__":
    main()
