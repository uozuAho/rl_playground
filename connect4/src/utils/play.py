from agents.agent import Agent
import env.connect4 as c4


def play_games_parallel(agent_1: Agent, agent_2: Agent, n_games: int):
    states = [c4.new_game() for _ in range(n_games)]
    dones = [False for _ in range(n_games)]
    winners: list[bool | None | c4.Player] = [False for _ in range(n_games)]
    turn = c4.PLAYER1
    while not all(dones):
        active_idx = [i for i in range(len(states)) if not dones[i]]
        active_envs = [states[i] for i in active_idx]
        agent = agent_1 if turn == c4.PLAYER1 else agent_2
        actions = agent.get_actions(active_envs)
        for i, action in zip(active_idx, actions):
            states[i] = c4.make_move(states[i], action)
            if states[i].done:
                dones[i] = True
                winners[i] = c4.calc_winner(states[i])
        turn = c4.other_player(turn)
    assert all(x is not False for x in winners)
    w = countif(winners, lambda x: x == c4.PLAYER1)
    ll = countif(winners, lambda x: x == c4.PLAYER2)
    d = countif(winners, lambda x: x is None)
    return w, ll, d


def countif(seq, pred):
    return sum(1 if pred(x) else 0 for x in seq)
