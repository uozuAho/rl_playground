from RLC.capture_chess.environment import Board
import chess
import typing as t


def play_random_game():
    board = Board()
    done = False
    # print(board.board)
    actions = []
    boards: t.List[chess.Board] = []
    rewards = []
    pieces_captured = []
    total_reward = 0
    while not done:
        # print('--------')
        action = board.get_random_action()
        boards.append(board.board.copy(stack=False))
        actions.append(action)
        # print(f'action: {action}')
        done, reward = board.step(action)
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
            # print(f"    {f} -> {t}, reward = {reward}")
        rewards.append(reward)
        # print(f'reward {reward}, total reward {total_reward}, board value {board.get_material_value()}')
        # print(board.board)
        # input("press a key...")
    return actions, boards, rewards, pieces_captured


for i in range(1):
    a, b, r, p = play_random_game()
    if sum(r) > 39:
        print(f'{len(r)} rewards. sum = {sum(r)}')
        print(','.join(str(int(x)) for x in r if x > 0))
        print('captured:', ','.join(str(x) for x in p))
        print(b[-1])
        break
