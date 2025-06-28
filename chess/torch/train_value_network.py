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
from tqdm import tqdm


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def pearsonr(x, y):
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


def generate_random_positions(n_positions=1000):
    """Generate random chess positions by playing random moves from the starting position."""
    positions = []

    for _ in range(n_positions):
        game = ChessGame()
        n_moves = random.randint(5, 50)  # Random game length

        for _ in range(n_moves):
            legal_moves = list(game.legal_moves())
            if not legal_moves or game.is_game_over():
                break
            move = random.choice(legal_moves)
            game.step(move)

        positions.append(game)

    return positions


def normalize_michniew_score(score):
    """Normalize Michniew score to [-1, 1] range similar to ValueNetwork output."""
    # Empirically observed range is roughly -3000 to 3000
    return np.tanh(score / 3000.0)


def generate_training_data(n_positions=5000):
    """Generate training data using random positions and Michniew evaluation."""
    print(f"Generating {n_positions} training positions...")
    positions = []
    targets = []

    for _ in tqdm(range(n_positions), desc="Generating positions"):
        game = ChessGame()
        n_moves = random.randint(5, 50)  # Random game length

        for _ in range(n_moves):
            legal_moves = list(game.legal_moves())
            if not legal_moves or game.is_game_over():
                break
            move = random.choice(legal_moves)
            game.step(move)

        # Get board state and evaluation
        state = game.state_np()
        board = chess.Board(game.fen())
        michniew_score = evaluate_board(board)
        normalized_score = normalize_michniew_score(michniew_score)

        positions.append(state)
        targets.append(normalized_score)

    return np.array(positions), np.array(targets)


def train_value_network(
    value_network,
    train_positions,
    train_targets,
    val_positions=None,
    val_targets=None,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    device="cpu",
):
    """Train the value network on the generated data."""
    value_network.to(device)
    value_network.train()

    optimizer = optim.Adam(value_network.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Convert to tensors
    train_positions = torch.tensor(train_positions, dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_targets, dtype=torch.float32).to(device)

    if val_positions is not None:
        val_positions = torch.tensor(val_positions, dtype=torch.float32).to(device)
        val_targets = torch.tensor(val_targets, dtype=torch.float32).to(device)

    train_losses = []
    val_losses = []

    print(f"Training value network for {epochs} epochs...")

    for epoch in tqdm(range(epochs), desc="Training"):
        # Shuffle data
        indices = torch.randperm(len(train_positions))
        epoch_losses = []

        # Training batches
        for i in range(0, len(train_positions), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_positions = train_positions[batch_indices]
            batch_targets = train_targets[batch_indices]

            optimizer.zero_grad()
            predictions = value_network(batch_positions).squeeze()
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)

        # Validation
        if val_positions is not None:
            value_network.eval()
            with torch.no_grad():
                val_predictions = value_network(val_positions).squeeze()
                val_loss = criterion(val_predictions, val_targets).item()
                val_losses.append(val_loss)
            value_network.train()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            if val_positions is not None:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

    return train_losses, val_losses


def compare_evaluations(value_network, positions, device="cpu"):
    """Compare ValueNetwork predictions with Michniew evaluate_board."""
    value_network.eval()

    network_scores = []
    michniew_scores = []

    with torch.no_grad():
        for game in positions:
            # Get board state for ValueNetwork
            state = game.state_np()
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            )

            # Get ValueNetwork prediction
            network_pred = value_network(state_tensor).item()
            network_scores.append(network_pred)

            # Get Michniew evaluation
            board = chess.Board(game.fen())
            michniew_pred = evaluate_board(board)
            michniew_normalized = normalize_michniew_score(michniew_pred)
            michniew_scores.append(michniew_normalized)

    return np.array(network_scores), np.array(michniew_scores)


def evaluate_accuracy(network_scores, michniew_scores):
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


def train_and_test_value_network():
    """Main function to train and test ValueNetwork accuracy against Michniew."""
    print("Training and testing ValueNetwork accuracy against Michniew evaluation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate training data
    train_positions, train_targets = generate_training_data(n_positions=5000)

    # Split into train/validation
    split_idx = int(0.8 * len(train_positions))
    val_positions = train_positions[split_idx:]
    val_targets = train_targets[split_idx:]
    train_positions = train_positions[:split_idx]
    train_targets = train_targets[:split_idx]

    print(f"Training set: {len(train_positions)} positions")
    print(f"Validation set: {len(val_positions)} positions")

    # Initialize and train ValueNetwork
    value_network = ValueNetwork()
    train_losses, val_losses = train_value_network(
        value_network,
        train_positions,
        train_targets,
        val_positions,
        val_targets,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        device=device,
    )

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Generate test positions for final evaluation
    print("\nGenerating test positions for final evaluation...")
    test_positions = generate_random_positions(1000)

    print(f"Evaluating {len(test_positions)} test positions...")
    network_scores, michniew_scores = compare_evaluations(
        value_network, test_positions, device
    )

    # Calculate metrics
    metrics = evaluate_accuracy(network_scores, michniew_scores)

    # Print results
    print("\n=== ACCURACY RESULTS ===")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"Pearson Correlation: {metrics['correlation']:.4f}")
    print(f"P-value: {metrics['p_value']:.6f}")

    # Interpretation
    print("\n=== INTERPRETATION ===")
    if metrics["correlation"] > 0.7:
        print(
            "✓ Strong positive correlation - ValueNetwork shows good agreement with Michniew"
        )
    elif metrics["correlation"] > 0.4:
        print(
            "⚠ Moderate correlation - ValueNetwork shows some agreement with Michniew"
        )
    else:
        print("✗ Poor correlation - ValueNetwork does not agree well with Michniew")

    if metrics["mae"] < 0.2:
        print(
            "✓ Low mean absolute error - ValueNetwork predictions are close to Michniew"
        )
    elif metrics["mae"] < 0.4:
        print(
            "⚠ Moderate mean absolute error - ValueNetwork predictions are somewhat close to Michniew"
        )
    else:
        print(
            "✗ High mean absolute error - ValueNetwork predictions differ significantly from Michniew"
        )

    # Create visualization
    plt.subplot(1, 2, 2)
    plt.scatter(michniew_scores, network_scores, alpha=0.6, s=20)
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
    train_and_test_value_network()
