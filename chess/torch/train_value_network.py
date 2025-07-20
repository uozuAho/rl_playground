from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import matplotlib.pyplot as plt
from lib.agents.greedy_agent import ValueNetwork
from lib.michniew import evaluate_board
from lib.env import ChessGame
import random
import typing as t
from tqdm import tqdm


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def pearsonr(x, y):
    """
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    0: no correlation
    +1,-1: perfect +/- correlation
    """
    x = np.array(x)
    y = np.array(y)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    sum_sq_x = np.sum((x - mean_x) ** 2)
    sum_sq_y = np.sum((y - mean_y) ** 2)

    denominator = np.sqrt(sum_sq_x * sum_sq_y)

    if denominator == 0:
        return 0.0, 1.0

    correlation = numerator / denominator

    n = len(x)
    if n <= 2:
        p_value = 1.0
    else:
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2 + 1e-8))
        p_value = 2 * (1 - np.abs(t_stat) / (np.abs(t_stat) + np.sqrt(n - 2)))
        p_value = np.clip(p_value, 0.0, 1.0)

    return correlation, p_value


def normalize_michniew_score(score: int):
    """Normalize Michniew score to [-1, 1] range similar to ValueNetwork output."""
    # Empirically observed range is roughly -3000 to 3000
    return np.tanh(score / 3000.0)


def generate_data(n_positions: int) -> t.Tuple[list[ChessGame], list[float]]:
    """returns [ChessGame], [normalised mich score]"""

    positions: list[ChessGame] = []
    scores: list[float] = []

    for _ in tqdm(range(n_positions), desc="Generating positions"):
        game = ChessGame()
        n_moves = random.randint(5, 50)  # Random game length

        for _ in range(n_moves):
            legal_moves = list(game.legal_moves())
            if not legal_moves or game.is_game_over():
                break
            move = random.choice(legal_moves)
            game.step(move)

        positions.append(game)
        scores.append(normalize_michniew_score(evaluate_board(game._board)))

    return positions, scores


def generate_data_file(n_positions: int, path: Path):
    with open(path, 'w') as ofile:
        for position, score in zip(*generate_data(n_positions)):
            ofile.write(f'{position.fen()}, {score}\n')


def read_datafile(path: Path):
    positions: list[ChessGame] = []
    scores: list[float] = []
    with open(path, 'r') as infile:
        for line in infile:
            fen, score = line.split(',')
            positions.append(ChessGame(fen))
            scores.append(float(score))
    return positions, scores


def train_value_network(
    value_network,
    train_positions: list[ChessGame],
    train_targets: list[float],
    test_positions: list[ChessGame],
    test_targets: list[float],
    epochs=100,
    batch_size=32,
    lr=1e-3,
    device: torch.device | str ="cpu",
):
    """Train the value network on the generated data."""
    value_network.to(device)
    value_network.train()

    optimizer = optim.Adam(value_network.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_positions_t = torch.tensor(np.array([b.state_np() for b in train_positions])).to(device)
    train_targets_t = torch.tensor(train_targets, dtype=torch.float32).to(device)
    test_positions_t = torch.tensor(np.array([b.state_np() for b in test_positions])).to(device)
    test_targets_t = torch.tensor(test_targets, dtype=torch.float32).to(device)

    train_losses = []
    val_losses = []

    print(f"Training value network for {epochs} epochs...")

    for epoch in tqdm(range(epochs), desc="Training"):
        # Shuffle data
        indices = torch.randperm(len(train_positions_t))
        epoch_losses = []

        # Training batches
        for i in range(0, len(train_positions_t), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_positions = train_positions_t[batch_indices]
            batch_targets = train_targets_t[batch_indices]

            optimizer.zero_grad()
            predictions = value_network(batch_positions).squeeze()
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)

        # Validation
        value_network.eval()
        with torch.no_grad():
            val_predictions = value_network(test_positions_t).squeeze()
            val_loss = criterion(val_predictions, test_targets_t).item()
            val_losses.append(val_loss)
        value_network.train()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            if test_positions_t is not None:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

    return train_losses, val_losses


def evaluate_accuracy(network_scores: np.ndarray, michniew_scores: np.ndarray):
    """Calculate various accuracy metrics."""
    mse = mean_squared_error(michniew_scores, network_scores)
    mae = mean_absolute_error(michniew_scores, network_scores)
    correlation, p_value = pearsonr(network_scores, michniew_scores)

    return {
        "mse": mse,
        "mae": mae,
        "correlation": correlation,
        "p_value": p_value,
        "rmse": np.sqrt(mse),
    }


def plot_comparison(network_scores, michniew_scores, metrics):
    """Create scatter plot comparing the two evaluation methods."""
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(michniew_scores, network_scores, alpha=0.6, s=20)
    plt.plot([-1, 1], [-1, 1], "r--", alpha=0.8, label="Perfect correlation")
    plt.xlabel("Michniew Evaluation (normalized)")
    plt.ylabel("ValueNetwork Prediction")
    plt.title(f"Evaluation Comparison\nCorrelation: {metrics['correlation']:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    residuals = network_scores - michniew_scores
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Residuals (Network - Michniew)")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution\nMAE: {metrics['mae']:.3f}")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def train_and_test_value_network(
        dataset_path: Path | None = None,
        dataset_size: int | None = None):
    print("Training and testing ValueNetwork accuracy against Michniew evaluation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if dataset_path:
        print(f"Reading data from {dataset_path}")
        positions, values = read_datafile(dataset_path)
    elif dataset_size:
        print("Generating data...")
        positions, values = generate_data(n_positions=dataset_size)
    else:
        raise Exception("Either gimme a dataset file or data size")

    split_idx = int(0.8 * len(positions))
    train_positions = positions[:split_idx]
    train_targets = values[:split_idx]
    test_positions = positions[split_idx:]
    test_targets = values[split_idx:]

    print(f"Training set: {len(train_positions)} positions")
    print(f"Validation set: {len(test_positions)} positions")

    value_network = ValueNetwork()
    train_losses, val_losses = train_value_network(
        value_network,
        train_positions,
        train_targets,
        test_positions,
        test_targets,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        device=device,
    )

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)

    print(f"Evaluating {len(test_positions)} test positions...")

    test_positions_t = torch.tensor([b.state_np() for b in test_positions]).to(device)
    value_network.eval()
    with torch.no_grad():
        network_scores_t: torch.Tensor = value_network(test_positions_t).squeeze()

    network_scores = network_scores_t.cpu().numpy()

    metrics = evaluate_accuracy(network_scores, np.array(test_targets))

    print("\n=== ACCURACY RESULTS ===")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"Pearson Correlation: {metrics['correlation']:.4f}")
    print(f"P-value: {metrics['p_value']:.6f}")

    plt.subplot(1, 2, 2)
    plt.scatter(test_targets, network_scores, alpha=0.6, s=20)
    plt.plot([-1, 1], [-1, 1], "r--", alpha=0.8, label="Perfect correlation")
    plt.xlabel("Michniew Evaluation (normalized)")
    plt.ylabel("ValueNetwork Prediction")
    plt.title(f"Evaluation Comparison\nCorrelation: {metrics['correlation']:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return value_network, metrics


if __name__ == "__main__":
    # generate_data_file(5000, Path('pymieches.csv'))
    train_and_test_value_network(Path('pymieches.csv'))
