import pytest
import chess
from utils.michniew import evaluate_board


@pytest.mark.parametrize(
    "fen,expected_score",
    [
        (chess.STARTING_FEN, 0),
        ("8/8/8/8/8/8/8/8 w - - 0 1", 0),
        ("8/8/8/8/8/8/8/K7 w - - 0 1", 20050),
        ("8/8/8/8/8/8/8/k7 w - - 0 1", -19950),
        ("8/8/8/8/8/8/8/4Q3 w - - 0 1", 895),
        ("8/8/8/8/8/8/8/4q3 w - - 0 1", -895),
        ("8/8/8/8/8/8/8/4P3 w - - 0 1", 100),
        ("8/8/8/8/8/8/8/4p3 w - - 0 1", -100),
        ("8/8/8/8/8/8/8/K6k w - - 0 1", 100),
        ("8/8/8/8/8/8/8/K5Qk w - - 0 1", 940),
        ("8/8/8/8/8/8/8/K5qk w - - 0 1", -840),
        ("8/8/8/8/8/8/8/K5k1 w - - 0 1", 90),
        # random
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 0),
        ("rnbq1bnr/ppppkppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQ - 3 4", 50),
        ("rnb2bnr/1p2p1p1/p1ppq2p/1P3p1k/1Q1PN3/P1P4P/3BPPP1/R3KBNR b KQ - 0 12", 110),
    ],
)
def test_evaluate_board(fen, expected_score):
    board = chess.Board(fen)
    score = evaluate_board(board)
    assert score == expected_score
