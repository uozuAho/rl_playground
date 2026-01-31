import env.connect4 as c4


def test_string_move():
    str1 = """.......
.......
.......
.......
.......
X......"""
    state1 = c4.from_string(str1)
    assert state1.current_player == c4.PLAYER2
    assert state1.winner is None
    assert not state1.done

    state2 = c4.make_move(state1, 0)
    assert (
        c4.to_string(state2)
        == """.......
.......
.......
.......
O......
X......"""
    )


def test_make_move():
    state = c4.new_game()

    state = c4.make_move(state, 3)
    assert state.board[5, 3] == c4.PLAYER1  # Should drop to bottom

    state = c4.make_move(state, 3)
    assert state.board[4, 3] == c4.PLAYER2  # Should stack on top


def test_check_winner():
    state = c4.from_string(""".......
.......
.......
.......
OOO....
XXX....""")

    state = c4.make_move(state, 3)
    assert state.done
    assert state.winner == c4.PLAYER1
    assert state.current_player == c4.PLAYER2


def test_string_conversion():
    state = c4.new_game()
    state = c4.make_move(state, 0)
    state = c4.make_move(state, 1)

    s = c4.to_string(state)
    restored = c4.from_string(s)

    assert c4.are_equal(state, restored)


def test_full_game():
    state = c4.new_game()

    while not state.done:
        valid_moves = c4.get_valid_moves(state)
        assert len(valid_moves) > 0
        move = valid_moves[0]
        assert c4.is_valid_move(state, move)
        state = c4.make_move(state, move)

    assert c4.calc_winner(state) is not None or c4.is_draw(state)
    assert state.done
    assert state.winner in [None, c4.PLAYER1, c4.PLAYER2]


def test_can_draw():
    state = c4.from_string("""XOXOXOO
XOXOXOX
OXOXOXO
OXOXOXO
XOXOXOX
XOXOXOX""")

    assert state.done
    assert state.winner is None
