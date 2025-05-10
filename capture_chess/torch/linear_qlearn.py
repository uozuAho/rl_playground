import torch

from lib.nets import LinearFCQNet
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
    policy_net = LinearFCQNet().to(device)
    target_net = LinearFCQNet().to(device)
    train(policy_net, target_net, n_episodes=1000, device=device, batch_size=64)


if __name__ == "__main__":
    main()
