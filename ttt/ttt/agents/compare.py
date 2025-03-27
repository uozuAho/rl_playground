import time
from collections import Counter
import typing as t

from ttt.agents.agent import TttAgent, TttAgent2
import ttt.env
import ttt.env2 as env2
from ttt.env import IllegalActionError, TicTacToeEnv


type GameResult = t.Literal['O', 'X', 'draw', 'X-illegal', 'O-illegal']


def play_game(agent_x: TttAgent, agent_o: TttAgent) -> GameResult:
    game = TicTacToeEnv()
    done = False
    while not done:
        if game.current_player == 'X':
            move = agent_x.get_action(game)
        else:
            move = agent_o.get_action(game)
        try:
            _, _, done, _, _ = game.step(move)
        except IllegalActionError:
            if game.current_player == 'X':
                return 'X-illegal'
            else:
                return 'O-illegal'
    status = game.get_status()
    if status == ttt.env.X_WIN:
        return 'X'
    elif status == ttt.env.O_WIN:
        return 'O'
    return 'draw'


def play_games(agent_x: TttAgent, agent_o: TttAgent, n_games: int):
    ctr: Counter[GameResult] = Counter()
    for _ in range(n_games):
        result = play_game(agent_x, agent_o)
        ctr[result] += 1
    return ctr


def play_and_report(
        agent_x: TttAgent,
        label_x: str,
        agent_o: TttAgent,
        label_o: str,
        n_games: int,
        quiet: bool = False):
    """ Plays N games and prints the results """
    start = time.time()
    results = play_games(agent_x, agent_o, n_games)
    end = time.time()

    x, o, d = results['X'], results['O'], results['draw']
    xi, oi = results['X-illegal'], results['O-illegal']
    xpc = 100 * results['X'] / n_games
    opc = 100 * results['O'] / n_games
    dpc = 100 * results['draw'] / n_games
    xipc = 100 * results['X-illegal'] / n_games
    oipc = 100 * results['O-illegal'] / n_games

    msg = (f'{label_x:<15} (x) vs {label_o:<15} (o). {n_games} games in {end-start:.1f}s. ' +
          f'x wins: {x:>3} ({xpc:>5.1f}%), o wins: {o:>3} ({opc:>5.1f}%), draws: {d:>3} ({dpc:>5.1f}%), ' +
          f'x illegal: {xi:>3} ({xipc:>5.1f}%), o illegal: {oi:>3} ({oipc:>5.1f}%)')
    if not quiet:
        print(msg)
    return msg


def play_game2(agent_x: TttAgent2, agent_o: TttAgent2) -> GameResult:
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
    ctr: Counter[GameResult] = Counter()
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
