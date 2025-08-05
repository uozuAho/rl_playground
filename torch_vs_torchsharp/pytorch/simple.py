import torch
import torch.nn as nn
import torch.optim as optim
import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def train_time_report(device):
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    X = torch.randn(10000, 100).to(device)
    y = torch.randint(0, 10, (10000,)).to(device)
    batch_size = 64
    batches_per_epoch = len(X) / batch_size
    num_epochs = 5

    print(f"{device=}")
    print(f"X shape: {X.shape}")
    print(f"len(x): {len(X)}")
    print(f"batch size: {batch_size}")

    def train():
        model.train()
        for _ in range(num_epochs):
            for i in range(0, len(X), batch_size):
                x_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                optimizer.zero_grad()
                output = model(x_batch)
                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()


    start = time.time()
    train()
    end = time.time()

    total_time = end - start
    epoch_avg = num_epochs / total_time
    batch_avg = (num_epochs * batches_per_epoch) / total_time
    sample_avg = (num_epochs * len(X)) / total_time

    print(f"Epoch time: {end - start:.4f} seconds")
    print(f"Epochs per second: {epoch_avg:.2f}")
    print(f"Batches per second: {batch_avg:.2f}")
    print(f"Samples per second: {sample_avg:.2f}")


train_time_report("cpu")
print()
train_time_report("cuda")
