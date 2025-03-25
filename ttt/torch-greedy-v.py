""" Greedy monte carlo learning agent. Learns to estimate state values.
    Plays a full game, then updates its model from the whole episode.
    Greedily plays the move that results in the best value estimate.

    Very slow. Doesn't improve, probably because the NN arch is too simple.
"""

from dataclasses import dataclass
import typing as t

import torch
import torch.nn as nn
import torch.optim as optim

from ttt.agents.agent import TttAgent
from ttt.agents.compare import play_and_report
from ttt.agents.random import RandomAgent
import ttt.env
from ttt.env import TicTacToeEnv


type Player = t.Literal['O', 'X']
type GameStatus = t.Literal['O', 'X', 'draw', 'in_progress']
type BoardState = list[int]


@dataclass
class Episode:
    states: list[BoardState]   # states in game order
    reward: int


class LinearFC(nn.Module):
    """ Fully connected linear/sequential NN """
    def __init__(self):
        super(LinearFC, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(9, 32, dtype=torch.float32),
            nn.Linear(32, 1, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor):
        x = self.stack(x)
        return x


class GreedyMcAgent(TttAgent):
    def __init__(self, nn: LinearFC, device: str):
        self.nn = nn
        self.device = device

    def get_action(self, env: TicTacToeEnv, learn=False):
        max_move = None
        max_val = -999
        for m in env.valid_actions():
            temp_env = env.copy()
            temp_env.step(m)
            val = self._nn_out(temp_env.board, learn)
            if val > max_val:
                max_move = m
                max_val = val
        return max_move

    def state_val(self, env: TicTacToeEnv, learn=False):
        return self._nn_out(env.board, learn)

    def _nn_out(self, state: BoardState, learn=False) -> float:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        if learn:
            return self.nn(state_t)
        else:
            with torch.no_grad():
                return self.nn(state_t)


def gamestatus(env: TicTacToeEnv) -> GameStatus:
    state = env.get_status()
    if state == ttt.env.O_WIN: return 'O'
    if state == ttt.env.X_WIN: return 'X'
    if state == ttt.env.DRAW: return 'draw'
    return 'in_progress'


def optimise_net_mc(
        value_net: LinearFC,
        episode: Episode,
        optimiser: optim.Optimizer,
        device: str,
        gamma=0.9):
    """ Monte carlo net update. Learns from a full episode """
    target = episode.reward
    for i in reversed(range(len(episode.states))):
        state = torch.tensor(episode.states[i], dtype=torch.float32).to(device)
        if i < len(episode.states) - 1:
            target = target * gamma

        optimiser.zero_grad()
        prediction = value_net(state)
        criterion = nn.MSELoss()
        loss = criterion(
            prediction,
            torch.tensor([target], dtype=torch.float32, device=device)
        )
        loss.backward()
        optimiser.step()


def play_game(agent_x: GreedyMcAgent, opponent_o: TttAgent):
    env = TicTacToeEnv()
    states=[]
    done = False
    while not done:
        state = env.board[:]
        states.append(env.board[:])
        if env.current_player == 'X':
            action = agent_x.get_action(env, learn=False)
        else:
            action = opponent_o.get_action(env)
        _, _, done, _, _ = env.step(action)
    result = gamestatus(env)
    reward = 1 if result == 'X' else 0 if result == 'draw' else -1
    return Episode(states, reward)


def train(
        agent_x: GreedyMcAgent,
        opponent_o: TttAgent,
        n_episodes: int,
        device: str,
        ):
    print(f"training for {n_episodes} episodes...")
    optimiser = optim.SGD(agent_x.nn.parameters(), lr=1e-4)
    for _ in range(n_episodes):
        episode = play_game(agent_x, opponent_o)
        optimise_net_mc(agent_x.nn, episode, optimiser, device)


def eval_agent(agent: GreedyMcAgent, opponent: TttAgent):
    play_and_report(agent, "mcts", opponent, "rando?", 100)


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = 'cpu'
print(f'using device {device}')
value_net = LinearFC().to(device)
agent = GreedyMcAgent(value_net, device)
opponent = RandomAgent()
eval_agent(agent, opponent)
print("PRESS CTRL-C TO STOP!")
for _ in range(1000):
    train(agent, opponent, 1000, device)
    eval_agent(agent, opponent)
