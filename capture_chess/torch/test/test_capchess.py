
import chess

from lib.env import CaptureChess


def play_random_game(action_limit):
    env = CaptureChess(action_limit)
    done = False
    actions = []
    boards: list[chess.Board] = []  # type: ignore
    rewards = []
    pieces_captured = []
    total_reward = 0
    while not done:
        action = env.get_random_action()
        boards.append(env.board.copy(stack=False))
        actions.append(action)
        done, reward = env.step(action)
        total_reward += reward
        if reward > 0:
            t = boards[-1].piece_at(action.to_square)
            if t:
                pieces_captured.append(t)
            # if not t:
            #     f = boards[-1].piece_at(action.from_square)
            #     print(boards[-1])
            #     print(action)
            #     print(f, t)
            #     print(reward)
            #     print(env.board)
            #     raise Exception("en passant?")
        rewards.append(reward)
    return actions, boards, rewards, pieces_captured


def test_play_random_games():
    action_limit = 25
    for i in range(50):
        actions, boards, rewards, pieces_captured = play_random_game(action_limit=action_limit)
        for x in rewards:
            # no promotion awards, only piece captures:
            assert x in [0,1,3,5,9]
            assert len(actions) <= action_limit
