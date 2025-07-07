import pytest
import chess
from lib.michniew import evaluate_board


@pytest.mark.parametrize(
    "fen,expected_sign",
    [
        (chess.STARTING_FEN, 0),  # Starting position, should be close to 0
        ("8/8/8/8/8/8/8/8 w - - 0 1", 0),  # Empty board
        ("8/8/8/8/8/8/8/K7 w - - 0 1", 1),  # Only white king
        ("8/8/8/8/8/8/8/k7 w - - 0 1", -1),  # Only black king
        ("8/8/8/8/8/8/8/4Q3 w - - 0 1", 1),  # Only white queen
        ("8/8/8/8/8/8/8/4q3 w - - 0 1", -1),  # Only black queen
        ("8/8/8/8/8/8/8/4P3 w - - 0 1", 1),  # Only white pawn
        ("8/8/8/8/8/8/8/4p3 w - - 0 1", -1),  # Only black pawn
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0),  # Symmetrical
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", 0),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0),
        ("8/8/8/8/8/8/8/8 w - - 0 1", 0),
    ],
)
def test_evaluate_board_sign(fen, expected_sign):
    board = chess.Board(fen)
    score = evaluate_board(board)
    if expected_sign == 0:
        assert abs(score) < 1e-3
    elif expected_sign > 0:
        assert score > 0
    else:
        assert score < 0


def test_evaluate_board_random_positions():
    # Some random positions
    fens = [
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "rnbq1bnr/ppppkppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQ - 3 4",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ]
    for fen in fens:
        board = chess.Board(fen)
        score = evaluate_board(board)
        # Just check that it returns a float and doesn't crash
        assert isinstance(score, (float, int))
