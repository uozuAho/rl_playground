from collections import Counter
import typing as t

from ttt.agents.agent import TttAgent, TttAgent2
import ttt.env
import ttt.env2 as env2
from ttt.env import TicTacToeEnv


type Result = t.Literal['O', 'X', 'draw']


def play_game(agent_x: TttAgent, agent_o: TttAgent) -> Result:
    """ Returns winner: X, O or draw """
    game = TicTacToeEnv()
    done = False
    while not done:
        if game.current_player == 'X':
            move = agent_x.get_action(game)
        else:
            move = agent_o.get_action(game)
        _, _, done, _, _ = game.step(move)
    status = game.get_status()
    if status == ttt.env.X_WIN:
        return 'X'
    elif status == ttt.env.O_WIN:
        return 'O'
    return 'draw'


def play_games(agent_x: TttAgent, agent_o: TttAgent, n_games: int) -> t.Tuple[int, int, int]:
    """ Returns counts: x wins, o wins, draws """
    ctr: Counter[Result] = Counter()
    for _ in range(n_games):
        winner = play_game(agent_x, agent_o)
        ctr[winner] += 1
    return ctr['X'], ctr['O'], ctr['draw']


def play_and_report(
        agent_x: TttAgent,
        label_x: str,
        agent_o: TttAgent,
        label_o: str,
        n_games: int):
    """ Plays N games and prints the results """
    x, o, d = play_games(agent_x, agent_o, n_games)
    xpc = 100 * x / n_games
    opc = 100 * o / n_games
    print(f'{label_x} (x) vs {label_o} (o). {n_games} games. ' +
          f'x wins: {x} ({xpc:.1f}%), o wins: {o} ({opc:.1f}%). draws: {d}')


def play_game2(agent_x: TttAgent2, agent_o: TttAgent2) -> Result:
    """ Returns winner: X, O or draw """
    env = env2.Env()
    done = False
    while not done:
        if env.current_player == env2.X:
            move = agent_x.get_action(env)
        else:
            move = agent_o.get_action(env)
        _, reward, done, _, _ = env.step(move)
    if reward == 1:
        return 'X'
    elif reward == -1:
        return 'O'
    return 'draw'


def play_games2(agent_x: TttAgent2, agent_o: TttAgent2, n_games: int) -> t.Tuple[int, int, int]:
    """ Returns counts: x wins, o wins, draws """
    ctr: Counter[Result] = Counter()
    for _ in range(n_games):
        winner = play_game2(agent_x, agent_o)
        ctr[winner] += 1
    return ctr['X'], ctr['O'], ctr['draw']


def play_and_report2(
        agent_x: TttAgent2,
        label_x: str,
        agent_o: TttAgent2,
        label_o: str,
        n_games: int):
    """ Plays N games and prints the results """
    x, o, d = play_games2(agent_x, agent_o, n_games)
    xpc = 100 * x / n_games
    opc = 100 * o / n_games
    print(f'{label_x} (x) vs {label_o} (o). {n_games} games. ' +
          f'x wins: {x} ({xpc:.1f}%), o wins: {o} ({opc:.1f}%). draws: {d}')
