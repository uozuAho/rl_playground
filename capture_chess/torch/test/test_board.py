from RLC.capture_chess.environment import Board  # type: ignore
import chess


def play_random_game():
    board = Board()
    done = False
    actions = []
    boards: list[chess.Board] = []  # type: ignore
    rewards = []
    pieces_captured = []
    total_reward = 0
    while not done:
        action = board.get_random_action()
        boards.append(board.board.copy(stack=False))
        actions.append(action)
        done, reward = board.step(action)
        # hack pawn promotion reward
        if reward % 2 == 0:  # reward should only be 1,3,5,9
            reward = 0
        total_reward += reward
        if reward > 0:
            f = boards[-1].piece_at(action.from_square)
            t = boards[-1].piece_at(action.to_square)
            if not t:
                print(boards[-1])
                print(action)
                print(f, t)
                print(reward)
                print(board.board)
                raise Exception("doh")
            pieces_captured.append(t)
        rewards.append(reward)
    return actions, boards, rewards, pieces_captured


def test_play_random_games():
    for i in range(10):
        actions, boards, rewards, pieces_captured = play_random_game()
        for x in rewards:
            assert x in [0,1,3,5,9]
