import numpy as np
from connect4 import (
    new_game, make_move, is_valid_move, check_winner, is_draw,
    get_valid_moves, state_to_string, string_to_state,
    PLAYER1, PLAYER2, ROWS, COLS
)


def test_new_game():
    state = new_game()
    assert state.shape == (ROWS, COLS)
    assert np.all(state == 0)


def test_make_move():
    state = new_game()

    state = make_move(state, 3, PLAYER1)
    assert state[5, 3] == PLAYER1  # Should drop to bottom

    state = make_move(state, 3, PLAYER2)
    assert state[4, 3] == PLAYER2  # Should stack on top


def test_check_winner():
    state = new_game()

    for col in range(4):
        state = make_move(state, col, PLAYER1)

    winner = check_winner(state)
    assert winner == PLAYER1


def test_string_conversion():
    state = new_game()
    state = make_move(state, 0, PLAYER1)
    state = make_move(state, 1, PLAYER2)

    s = state_to_string(state)
    restored = string_to_state(s)

    assert np.array_equal(state, restored)


def test_game_flow():
    state = new_game()

    valid_moves = get_valid_moves(state)
    assert len(valid_moves) == COLS

    state = make_move(state, 3, PLAYER1)
    state = make_move(state, 3, PLAYER2)

    assert is_valid_move(state, 3)
    assert check_winner(state) is None
