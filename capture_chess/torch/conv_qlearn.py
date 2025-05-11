from pathlib import Path
import time
import matplotlib.pyplot as plt

from lib import torch_utils
from lib.evaluator import evaluate
from lib.nets import ConvQNet
import lib.trainer


TRAINED_MODEL_PATH = Path("./trained_models/convq2")
FORCE_TRAIN = True


def main():
    device = torch_utils.find_device()
    if not FORCE_TRAIN and TRAINED_MODEL_PATH.exists():
        policy_net = ConvQNet.load(TRAINED_MODEL_PATH, device)
    else:
        policy_net = train(device)
    evaluate("ConvQNet", policy_net, 10, device)


def train(device):
    print(f"Using device: {device}")
    policy_net = ConvQNet().to(device)
    target_net = ConvQNet().to(device)

    def interim_eval(ep_num: int):
        if ep_num > 64 and ep_num % 50 == 0:
            try:
                evaluate("adsf", policy_net, 5, device)
            except Exception as e:
                print(f"nonfatal: eval error: {e}")

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
    losses, rewards = lib.trainer.train(
        policy_net, target_net, device=device, ep_callback=interim_eval, **params
    )
    policy_net.save(TRAINED_MODEL_PATH)

    print(f"trained for {time.time() - start:0.1f}s")

    x = list(range(len(losses)))
    plt.xlabel("episodes")
    plt.plot(x, losses, label="loss")
    plt.plot(x, rewards, label="reward")
    plt.legend()
    plt.show()

    return policy_net


if __name__ == "__main__":
    main()
