import torch

from nets.nets import ConvQNet
from training.trainer import train


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
    # todo: show net shape
    train(policy_net, target_net, n_episodes=1000, device=device, batch_size=64)


if __name__ == "__main__":
    main()
