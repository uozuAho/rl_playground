import numpy as np
import env.connect4 as c4


def test_new_game():
    state = c4.new_game()
    assert np.all(state == 0)


def test_make_move():
    state = c4.new_game()

    state = c4.make_move(state, 3, c4.PLAYER1)
    assert state[5, 3] == c4.PLAYER1  # Should drop to bottom

    state = c4.make_move(state, 3, c4.PLAYER2)
    assert state[4, 3] == c4.PLAYER2  # Should stack on top


def test_check_winner():
    state = c4.new_game()

    for col in range(4):
        state = c4.make_move(state, col, c4.PLAYER1)

    winner = c4.winner(state)
    assert winner == c4.PLAYER1


def test_string_conversion():
    state = c4.new_game()
    state = c4.make_move(state, 0, c4.PLAYER1)
    state = c4.make_move(state, 1, c4.PLAYER2)

    s = c4.to_string(state)
    restored = c4.from_string(s)

    assert np.array_equal(state, restored)


def test_full_game():
    state = c4.new_game()
    turn = c4.PLAYER1

    while not c4.is_terminal(state):
        valid_moves = c4.get_valid_moves(state)
        assert len(valid_moves) > 0
        move = valid_moves[0]
        assert c4.is_valid_move(state, move)
        state = c4.make_move(state, move, turn)
        turn = c4.other_player(turn)

    assert c4.winner(state) is not None or c4.is_draw(state)
