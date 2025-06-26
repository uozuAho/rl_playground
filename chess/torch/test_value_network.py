import torch
import numpy as np
import chess
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from lib.greedy_agent import ValueNetwork
from lib.michniew import evaluate_board
from lib.env import ChessGame
import random


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


def compare_evaluations(value_network, positions, device='cpu'):
    """Compare ValueNetwork predictions with Michniew evaluate_board."""
    value_network.eval()

    network_scores = []
    michniew_scores = []

    with torch.no_grad():
        for game in positions:
            # Get board state for ValueNetwork
            state = game.state_np()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

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
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'p_value': p_value,
        'rmse': np.sqrt(mse)
    }


def plot_comparison(network_scores, michniew_scores, metrics):
    """Create scatter plot comparing the two evaluation methods."""
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(michniew_scores, network_scores, alpha=0.6, s=20)
    plt.plot([-1, 1], [-1, 1], 'r--', alpha=0.8, label='Perfect correlation')
    plt.xlabel('Michniew Evaluation (normalized)')
    plt.ylabel('ValueNetwork Prediction')
    plt.title(f'Evaluation Comparison\nCorrelation: {metrics["correlation"]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    residuals = network_scores - michniew_scores
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (Network - Michniew)')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution\nMAE: {metrics["mae"]:.3f}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def test_value_network_accuracy():
    """Main test function to evaluate ValueNetwork accuracy against Michniew."""
    print("Testing ValueNetwork accuracy against Michniew evaluation...")

    # Initialize ValueNetwork
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value_network = ValueNetwork().to(device)

    # Generate test positions
    print("Generating random chess positions...")
    positions = generate_random_positions(1000)

    print(f"Evaluating {len(positions)} positions...")
    network_scores, michniew_scores = compare_evaluations(value_network, positions, device)

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
    if metrics['correlation'] > 0.7:
        print("✓ Strong positive correlation - ValueNetwork shows good agreement with Michniew")
    elif metrics['correlation'] > 0.4:
        print("⚠ Moderate correlation - ValueNetwork shows some agreement with Michniew")
    else:
        print("✗ Poor correlation - ValueNetwork does not agree well with Michniew")

    if metrics['mae'] < 0.2:
        print("✓ Low mean absolute error - ValueNetwork predictions are close to Michniew")
    elif metrics['mae'] < 0.4:
        print("⚠ Moderate mean absolute error - ValueNetwork predictions are somewhat close to Michniew")
    else:
        print("✗ High mean absolute error - ValueNetwork predictions differ significantly from Michniew")

    # Create visualization
    plot_comparison(network_scores, michniew_scores, metrics)

    return metrics


if __name__ == "__main__":
    test_value_network_accuracy()
