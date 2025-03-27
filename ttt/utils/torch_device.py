import torch


def find_device(override: str | None = None):
    device = override if override else (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f'using device {device}')
    return device