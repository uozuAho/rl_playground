"""
Training on capture chess from https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-3-q-networks

Capture chess rules:
- max 25 moves
- agent plays white
- opponent is part of the environment and makes random moves
    - see https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/environment.py#L85
- rewards for capturing pieces
    - no negative reward for losing pieces
- reward at end of episode = 0

Code:

My summary of the code from above: https://github.com/arjangroen/RLC

board = Board()
agent = Agent(network='conv',gamma=0.1,lr=0.07)
R = Q_learning(agent,board)
R.agent.fix_model()   # saves the network state to separate storage
pgn = R.learn(iters=750)

learn: https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/learn.py#L26
    - play_game N games, N = iters
    - fix_model each 10 games:  fixed_model is the target network

play_game: https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/learn.py#L53
    - play N games
    - each move
        - save to replay memory: [state, (move from, move to), reward, next_state]
        - remove old moves is memory is full
        - add 1 to sampling_probs   # todo: what's this?
        - update_agent

update_agent: https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/learn.py#L139C9-L139C21
    - sample replay memory
    - self.agent.network_udpate: update model with sample
    - update sampling_probs with returned td errors  # todo: why?

network_udpate: https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/agent.py#L111
    - update the model
"""

from collections import deque
from dataclasses import dataclass
import typing as t
import random
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from RLC.capture_chess.environment import Board


@dataclass
class Transition:
    state: np.ndarray
    move: tuple[int, int]  # move from, to
    next_state: np.ndarray
    reward: float


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class LinearModel(nn.Module):
    def __init__(self, device):
        super(LinearModel, self).__init__()
        self.flatten = nn.Flatten()
        # 8*8*8 = 8 layers of 8x8 boards, one layer per piece type
        # 64*64 = move piece from X (64 options) to Y (64 options)
        # eg output indexes:
        # 0: move 0 to 0
        # 1: move 0 to 1
        # ...
        # 4094: move 63 to 62
        # 4095: move 63 to 63
        self.stack = nn.Sequential(
            nn.Linear(8*8*8, 64*64, dtype=torch.float64)
        )
        self.device = device

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.stack(x)
        return x

    def get_action(self, board: Board) -> chess.Move:
        nn_input = torch.from_numpy(board.layer_board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            nn_output = self(nn_input)
        action_values = torch.reshape(nn_output, (64, 64))
        legal_mask = torch.from_numpy(board.project_legal_moves()).to(self.device)
        action_values = torch.multiply(action_values, legal_mask)
        move_from = torch.argmax(action_values) // 64
        move_to = torch.argmax(action_values) % 64
        moves = [x for x in board.board.generate_legal_moves()
                 if x.from_square == move_from and x.to_square == move_to]
        if len(moves) == 0:
            # If all legal moves have negative action value, explore
            move = board.get_random_action()
            move_from = move.from_square
            move_to = move.to_square
        else:
            # If there are multiple max-moves, pick a random one
            move = random.choice(moves)
        return move


def play_game(model: LinearModel, board: Board):
    done = False
    i = 0
    while not done:
        action = model.get_action(board)  # todo: need with torch.no_grad()?
        done, reward = board.step(action)
        i += 1
        if i > 200:
            raise "too many actions"
    print("done")


def show_board_model_shapes_types():
    board = Board()
    print(board.board)
    # each piece type is on a different layer
    print(board.layer_board[0,::-1,:].astype(int))
    print(board.layer_board.shape)
    print(torch.from_numpy(board.layer_board).shape)
    print(nn.Flatten(0)(torch.from_numpy(board.layer_board)).shape)
    action = model.get_action(board)
    print(action)


def is_endstate(layer_board: np.ndarray):
    return np.array_equal(layer_board, layer_board * 0)


def optimise_model(
        model: LinearModel,
        transitions: t.List[Transition],
        optimiser: optim.Optimizer,
        device: str,
        gamma=0.99):
    """ Update model weights using a batch of transitions. Returns loss """

    batch_size = len(transitions)
    states = torch.stack([torch.from_numpy(t.state) for t in transitions]).to(device)
    moves = [(t.move[0] * 64 + t.move[1]) for t in transitions]
    moves = torch.tensor(moves, dtype=torch.long, device=device)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float64, device=device)
    next_states = torch.stack([torch.from_numpy(t.next_state) for t in transitions]).to(device)

    qs = model(states)
    qsa = qs.gather(1, moves.unsqueeze(1)).squeeze(1)

    non_final_mask = [not is_endstate(t.next_state) for t in transitions]
    non_final_mask = torch.tensor(non_final_mask, dtype=torch.bool, device=device)
    non_final_next_states = next_states[non_final_mask]
    max_qnext = torch.zeros(batch_size, dtype=torch.float64, device=device)

    if len(non_final_next_states) > 0:
        with torch.no_grad():
            max_qnext[non_final_mask] = model(non_final_next_states).max(1)[0]

    exp_qsa = rewards + gamma * max_qnext

    criterion = nn.MSELoss()
    loss = criterion(qsa, exp_qsa)

    optimiser.zero_grad()
    loss.backward()
    # gradient clip/limit
    torch.nn.utils.clip_grad_value_(model.parameters(), 100)
    optimiser.step()

    return loss.item()


def train(
        model: LinearModel,
        n_episodes: int,
        device: str,
        n_episode_action_limit=25,
        batch_size=32,
        ):
    board = Board()
    optimiser = optim.SGD(model.parameters(), lr=1e-4)
    episode = 0
    ep_losses = []
    ep_rewards = []
    eps = 0.1  # todo: epsilon schedule
    replay_mem = ReplayMemory(1000)
    for ep in range(n_episodes):
        print(f'{ep}/{n_episodes}')
        losses = []
        rewards = []
        board.reset()
        game_over = False
        while not game_over and len(rewards) < n_episode_action_limit:
            state = board.layer_board
            if np.random.uniform(0, 1) < eps:
                action = board.get_random_action()
            else:
                action = model.get_action(board)
            game_over, reward = board.step(action)
            # hack pawn promotion reward (don't want reward for pawn promotion)
            if reward % 2 == 0:  # reward should only be 1,3,5,9
                reward = 0
            next_state = board.layer_board
            rewards.append(reward)
            episode += 1

            replay_mem.push(Transition(
                state,
                (action.from_square, action.to_square),
                next_state,
                reward
            ))

            if len(replay_mem) >= batch_size:
                loss = optimise_model(
                    model,
                    replay_mem.sample(batch_size), optimiser, device)
                losses.append(loss)
        ep_losses.append(sum(losses))
        ep_rewards.append(sum(rewards))
    episode = list(range(n_episodes))
    plt.xlabel('episodes')
    plt.plot(episode, ep_losses, label='loss')
    plt.plot(episode, ep_rewards, label='reward')
    plt.legend()
    plt.show()


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
# device = 'cpu'
print(f'Using device: {device}')
# show_board_model_shapes_types()
board = Board()
model = LinearModel(device).to(device)
# play_game(model, board)
train(model, 10, device, batch_size=64)
