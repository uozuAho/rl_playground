import pytest
import chess
from lib.michniew import evaluate_board


@pytest.mark.parametrize(
    "fen,expected_score",
    [
        (chess.STARTING_FEN, 0),
        ("8/8/8/8/8/8/8/8 w - - 0 1", 0),  # Empty board
        ("8/8/8/8/8/8/8/K7 w - - 0 1", 20050),  # Only white king
        ("8/8/8/8/8/8/8/k7 w - - 0 1", -19950),  # Only black king
        ("8/8/8/8/8/8/8/4Q3 w - - 0 1", 895),  # Only white queen
        ("8/8/8/8/8/8/8/4q3 w - - 0 1", -895),  # Only black queen
        ("8/8/8/8/8/8/8/4P3 w - - 0 1", 100),  # Only white pawn
        ("8/8/8/8/8/8/8/4p3 w - - 0 1", -100),  # Only black pawn
        # random
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 0),
        ("rnbq1bnr/ppppkppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQ - 3 4", 50),
    ],
)
def test_evaluate_board(fen, expected_score):
    board = chess.Board(fen)
    score = evaluate_board(board)
    assert score == expected_score
