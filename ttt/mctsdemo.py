from collections import Counter
from ttt.agents.agent import TttAgent
from ttt.agents.mcts import MctsAgent
from ttt.agents.perfect import PerfectAgent
from ttt.agents.random import RandomAgent
import ttt.env
from ttt.env import TicTacToeEnv


def play_game(agent_x: TttAgent, agent_o: TttAgent):
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


def play_games(agent_x: TttAgent, agent_o: TttAgent, n_games: int):
    """ Returns counts: x wins, o wins, draws """
    ctr = Counter()
    for i in range(n_games):
        winner = play_game(agent_x, agent_o)
        ctr[winner] += 1
    return ctr['X'], ctr['O'], ctr['draw']


def play_and_report(
        agent_x: TttAgent,
        label_x: str,
        agent_o: TttAgent,
        label_o: str,
        n_games: int):
    x, o, d = play_games(agent_x, agent_o, n_games)
    xpc = 100 * x / n_games
    opc = 100 * o / n_games
    print(f'{label_x} (x) vs {label_o} (o). {n_games} games. ' +
          f'x wins: {x} ({xpc:.1f}%), o wins: {o} ({opc:.1f}%). draws: {d}')


play_and_report(RandomAgent(), "rando", RandomAgent(), "rando", 1000)       # x wins 60%
play_and_report(MctsAgent(n_sims=1), "mcts1", RandomAgent(), "rando", 50)   # tiny bit better
play_and_report(MctsAgent(n_sims=5), "mcts5", RandomAgent(), "rando", 50)   # ~88%
play_and_report(MctsAgent(n_sims=10), "mcts10", RandomAgent(), "rando", 50) # ~94%
play_and_report(RandomAgent(), "rando", MctsAgent(n_sims=10), "mcts10", 50)   # o wins ~68%
play_and_report(RandomAgent(), "rando", MctsAgent(n_sims=50), "mcts50", 50)   # o wins ~90%
play_and_report(MctsAgent(n_sims=10), "mcts10", PerfectAgent('O'), "perfect", 50)  # o wins ~50%
play_and_report(MctsAgent(n_sims=100), "mcts100", PerfectAgent('O'), "perfect", 50) # o wins ~34%
play_and_report(MctsAgent(n_sims=200), "mcts200", PerfectAgent('O'), "perfect", 50) # Slow! o wins ~32%
