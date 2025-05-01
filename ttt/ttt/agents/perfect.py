from ttt.agents.agent import TttAgent
import ttt.env as t3


class PerfectAgent(TttAgent):
    def get_action(self, env: t3.Env):
        return find_best_move(env, env.current_player)


def player_won(env: t3.Env, player: t3.Player):
    return t3.winner(env.board) == player


def find_best_move(env: t3.Env, player=t3.X):
    opponent = t3.other_player(player)

    # make a winning move
    for pos in range(9):
        if env.board[pos] == t3.EMPTY:
            tempboard = env.board[:]
            tempboard[pos] = player
            status = t3.status(tempboard)
            if status == t3.O and player == t3.O:
                return pos
            if status == t3.X and player == t3.X:
                return pos

    # block opponent's win
    for pos in range(9):
        if env.board[pos] == t3.EMPTY:
            tempboard = env.board[:]
            tempboard[pos] = opponent
            status = t3.status(tempboard)
            if status == t3.O and player == t3.X:
                return pos
            if status == t3.X and player == t3.O:
                return pos

    num_empty = sum(1 for x in env.board if x == 0)
    if player == t3.O and num_empty == 6:
        x = t3.X
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
        # block another winning setup for x:
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
        # block yet another winning setup for x:
        # 1. x goes side
        # 2. o goes middle
        # 3. x goes adjacent side
        # 4. o should take any corner except the one opposite the corner between
        #    the two x's
        if env.board[1] == x:
            if env.board[3] == x:
                return 0
            if env.board[5] == x:
                return 2
        if env.board[3] == x:
            if env.board[1] == x:
                return 0
            if env.board[7] == x:
                return 6
        if env.board[5] == x:
            if env.board[1] == x:
                return 2
            if env.board[7] == x:
                return 8
        if env.board[7] == x:
            if env.board[5] == x:
                return 8
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
