import time
from collections import Counter
import typing as t

from agents.agent import TttAgent
import ttt.env as t3


type GameResult = t.Literal["O", "X", "draw", "X-illegal", "O-illegal"]


def play_game(agent_x: TttAgent, agent_o: TttAgent) -> GameResult:
    game = t3.TttEnv()
    done = False
    while not done:
        if game.current_player == t3.X:
            move = agent_x.get_action(game)
        else:
            move = agent_o.get_action(game)
        try:
            _, reward, done, _, _ = game.step(move)
        except t3.IllegalActionError:
            if game.current_player == t3.X:
                return "X-illegal"
            else:
                return "O-illegal"
    if reward == 1:
        return "X"
    elif reward == -1:
        return "O"
    return "draw"


def play_games(agent_x: TttAgent, agent_o: TttAgent, n_games: int):
    ctr: Counter[GameResult] = Counter()
    for _ in range(n_games):
        result = play_game(agent_x, agent_o)
        ctr[result] += 1
    return ctr


def play_games_parallel(agent_x: TttAgent, agent_o: TttAgent, n_games: int):
    envs = [t3.TttEnv() for _ in range(n_games)]
    done = [False for _ in range(n_games)]
    outcomes = [None for _ in range(n_games)]
    turn = t3.X
    while not all(done):
        active_idx = [i for i in range(len(envs)) if not done[i]]
        active_envs = [envs[i] for i in active_idx]
        agent = agent_x if turn == t3.X else agent_o
        actions = agent.get_actions(active_envs)
        results = [env.step(a) for env, a in zip(active_envs, actions)]
        for i, result in zip(active_idx, results):
            _, reward, terminated, *_ = result
            if terminated:
                done[i] = True
                outcomes[i] = reward
        turn = t3.other_player(turn)
    assert all(x is not None for x in outcomes)
    return Counter("X" if o == 1 else "O" if o == -1 else "draw" for o in outcomes)


def play_and_report(
    agent_x: TttAgent,
    label_x: str,
    agent_o: TttAgent,
    label_o: str,
    n_games: int,
    quiet: bool = False,
    show_illegal: bool = False,
):
    start = time.time()
    results = play_games(agent_x, agent_o, n_games)
    end = time.time()

    x, o, d = results["X"], results["O"], results["draw"]
    xi, oi = results["X-illegal"], results["O-illegal"]
    xpc = 100 * results["X"] / n_games
    opc = 100 * results["O"] / n_games
    dpc = 100 * results["draw"] / n_games
    xipc = 100 * results["X-illegal"] / n_games
    oipc = 100 * results["O-illegal"] / n_games

    label_width = max(len(x) for x in (label_x, label_o)) + 2

    msg = (
        f"{label_x.rjust(label_width)} (x) vs {label_o.rjust(label_width)} (o). {n_games} games in {end - start:.1f}s. "
        + f"x wins: {x:>3} ({xpc:>5.1f}%), o wins: {o:>3} ({opc:>5.1f}%), draws: {d:>3} ({dpc:>5.1f}%), "
        + f"x illegal: {xi:>3} ({xipc:>5.1f}%), o illegal: {oi:>3} ({oipc:>5.1f}%)"
    )

    if show_illegal:
        msg += (
            f", x illegal: {xi:>3} ({xipc:>5.1f}%), o illegal: {oi:>3} ({oipc:>5.1f}%)"
        )

    if not quiet:
        print(msg)
    return msg
