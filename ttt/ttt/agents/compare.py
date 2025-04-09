import time
from collections import Counter
import typing as t

from ttt.agents.agent import TttAgent
import ttt.env as env


type GameResult = t.Literal['O', 'X', 'draw', 'X-illegal', 'O-illegal']


def play_game(agent_x: TttAgent, agent_o: TttAgent) -> GameResult:
    game = env.Env()
    done = False
    while not done:
        if game.current_player == env.X:
            move = agent_x.get_action(game)
        else:
            move = agent_o.get_action(game)
        try:
            _, reward, done, _, _ = game.step(move)
        except env.IllegalActionError:
            if game.current_player == env.X:
                return 'X-illegal'
            else:
                return 'O-illegal'
    if reward == 1:
        return 'X'
    elif reward == -1:
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

    msg = (f'{label_x:<16} (x) vs {label_o:<16} (o). {n_games} games in {end-start:.1f}s. ' +
          f'x wins: {x:>3} ({xpc:>5.1f}%), o wins: {o:>3} ({opc:>5.1f}%), draws: {d:>3} ({dpc:>5.1f}%), ' +
          f'x illegal: {xi:>3} ({xipc:>5.1f}%), o illegal: {oi:>3} ({oipc:>5.1f}%)')
    if not quiet:
        print(msg)
    return msg
