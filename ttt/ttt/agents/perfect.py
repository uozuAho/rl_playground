from ttt.agents.agent import TttAgent
import ttt.env
from ttt.env import TicTacToeEnv


class PerfectAgent(TttAgent):
    def get_action(self, env: TicTacToeEnv):
        return find_best_move(env, env.current_player)


def player_won(env, player):
    if player == 'X' and env.get_status() == ttt.env.X_WIN:
        return True
    elif player == 'O' and env.get_status() == ttt.env.O_WIN:
        return True
    return False


def find_best_move(env: TicTacToeEnv, player='X'):
    opponent = 'O' if player == 'X' else 'X'

    # Winning move
    # hacking env internals, using env.copy/env.step is dangerous
    for pos in range(9):
        if env.board[pos] == ttt.env.EMPTY_CODE:
            tempboard = env.board[:]
            tempboard[pos] = ttt.env.tocode(player)
            status = ttt.env.check_game_status(tempboard)
            if status == ttt.env.O_WIN and player == 'O':
                return pos
            if status == ttt.env.X_WIN and player == 'X':
                return pos

    # Block opponent's win:
    # hacking this for now. hard to do esp. with an opponent in place
    for pos in range(9):
        if env.board[pos] == ttt.env.EMPTY_CODE:
            tempboard = env.board[:]
            tempboard[pos] = ttt.env.tocode(opponent)
            status = ttt.env.check_game_status(tempboard)
            if status == ttt.env.O_WIN and player == 'X':
                return pos
            if status == ttt.env.X_WIN and player == 'O':
                return pos

    num_empty = sum(1 for x in env.board if x == 0)
    if player == 'O' and num_empty == 6:
        x = ttt.env.X_CODE
        # block winning setup for x:
        # 1. x goes corner
        # 2. o goes middle
        # 3. x goes opposite corner
        # 4. o needs to take a side instead of corner to prevent x win
        if (
            (env.board[0] == x and env.board[8] == x)
            or
            (env.board[2] == x and env.board[6] == x)
        ):
            for a in env.valid_actions():
                if a in [1, 3, 5, 7]:
                    return a
        # todo: block another winning setup for x
        # 1. x goes corner
        # 2. o goes middle
        # 3. x goes opposite side
        # 4. o needs to take corner 'between' the 2 xs to prevent x win
        if env.board[0] == x:
            if env.board[5] == x:
                return 2
            if env.board[7] == x:
                return 6
        if env.board[2] == x:
            if env.board[3] == x:
                return 0
            if env.board[7] == x:
                return 8
        if env.board[6] == x:
            if env.board[1] == x:
                return 0
            if env.board[5] == x:
                return 8
        if env.board[8] == x:
            if env.board[1] == x:
                return 2
            if env.board[3] == x:
                return 6

    # Prioritize center
    if 4 in env.valid_actions(): return 4

    # Take a corner
    for a in env.valid_actions():
        if a in [0, 2, 6, 8]:
            return a

    # Take a side
    for a in env.valid_actions():
        if a in [1, 3, 5, 7]:
            return a

    raise Exception("Can't find a move to play!")
