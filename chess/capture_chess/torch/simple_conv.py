"""
Training on capture chess from https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-3-q-networks

Linear & Convolution nets are implemented. Conv is faster as it reduces the
input layer dimensionality.

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
import math
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


class LinearFC(nn.Module):
    """ Fully connected linear/sequential NN """
    def __init__(self):
        super(LinearFC, self).__init__()
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

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.stack(x)
        return x


class ConvNet(nn.Module):
    """ Dunno if this is correct, I just got chat gpt to convert from TF to torch:
        https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/agent.py#L73

        This trains much quicker than the liner FC net, as the conv layers reduce
        dimensionality by 8 (I think)
    """
    def __init__(self):
        super(ConvNet, self).__init__()

        # 1x1 conv layers used to blend input layers
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, dtype=torch.float64)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, dtype=torch.float64)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1_flat = x1.view(x1.size(0), 64, 1)
        x2_flat = x2.view(x2.size(0), 1, 64)
        output = torch.bmm(x1_flat, x2_flat)
        output = output.view(output.size(0), -1)
        return output


def get_nn_move(net: nn.Module, board: Board, device) -> chess.Move:
    """ Assumes a net with a 1x4096 (64x64) output, which represents a
        move from (64) -> to (64)
    """
    nn_input = torch.from_numpy(board.layer_board).unsqueeze(0).to(device)
    with torch.no_grad():
        nn_output = net(nn_input)
    action_values = torch.reshape(nn_output, (64, 64))
    legal_mask = torch.from_numpy(board.project_legal_moves()).to(device)
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


def play_game(net: LinearFC, board: Board):
    done = False
    i = 0
    while not done:
        with torch.no_grad:
            action = get_nn_move(net, board, 'cpu')
        done, reward = board.step(action)
        i += 1
        if i > 200:
            raise "too many actions"
    print("done")


def show_board_net_shapes_types():
    board = Board()
    net = LinearFC('cpu')
    print(board.board)
    # each piece type is on a different layer
    print(board.layer_board[0,::-1,:].astype(int))
    print(board.layer_board.shape)
    print(torch.from_numpy(board.layer_board).shape)
    print(nn.Flatten(0)(torch.from_numpy(board.layer_board)).shape)
    action = get_nn_move(net, board, 'cpu')
    print(action)


def is_endstate(layer_board: np.ndarray):
    return np.array_equal(layer_board, layer_board * 0)


def optimise_net(
        policy_net: LinearFC,
        target_net: LinearFC,
        transitions: t.List[Transition],
        optimiser: optim.Optimizer,
        device: str,
        gamma=0.99):
    """ Update NN weights using a batch of transitions. Returns loss """

    batch_size = len(transitions)
    states = torch.stack([torch.from_numpy(t.state) for t in transitions]).to(device)
    moves = [(t.move[0] * 64 + t.move[1]) for t in transitions]
    moves = torch.tensor(moves, dtype=torch.long, device=device)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float64, device=device)
    next_states = torch.stack([torch.from_numpy(t.next_state) for t in transitions]).to(device)

    qs = policy_net(states)
    qsa = qs.gather(1, moves.unsqueeze(1)).squeeze(1)

    non_final_mask = [not is_endstate(t.next_state) for t in transitions]
    non_final_mask = torch.tensor(non_final_mask, dtype=torch.bool, device=device)
    non_final_next_states = next_states[non_final_mask]
    max_qnext = torch.zeros(batch_size, dtype=torch.float64, device=device)

    if len(non_final_next_states) > 0:
        with torch.no_grad():
            max_qnext[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    exp_qsa = rewards + gamma * max_qnext

    criterion = nn.MSELoss()
    loss = criterion(qsa, exp_qsa)

    optimiser.zero_grad()
    loss.backward()
    # gradient clip/limit
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimiser.step()

    return loss.item()


def update_target(policy_net: LinearFC, target_net: LinearFC, tau=1.0):
    """ (Soft) Update the target network. Tau = hardness. If Tau = 1.0, it's a
        hard update, ie the target net is set to equal the policy net.
    """
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = \
            policy_net_state_dict[key]*tau + \
            target_net_state_dict[key]*(1-tau)
    target_net.load_state_dict(target_net_state_dict)


def epsilon(eps_start, eps_end, n_total_ep, ep):
    return eps_end + (eps_start - eps_end) * math.exp(-6. * ep / n_total_ep)


def train(
        policy_net: LinearFC,
        target_net: LinearFC,
        n_episodes: int,
        device: str,
        n_episode_action_limit=25,
        batch_size=32,
        target_net_update_eps=10,
        target_net_update_tau=1.0
        ):
    board = Board()
    optimiser = optim.SGD(policy_net.parameters(), lr=1e-4)
    episode = 0
    ep_losses = []
    ep_rewards = []
    replay_mem = ReplayMemory(1000)
    for ep in range(n_episodes):
        print(f'{ep}/{n_episodes}')
        if ep % target_net_update_eps == 0:
            update_target(policy_net, target_net, target_net_update_tau)
        losses = []
        rewards = []
        board.reset()
        game_over = False
        eps = epsilon(.99, .01, n_episodes, ep)
        while not game_over and len(rewards) < n_episode_action_limit:
            state = board.layer_board
            if np.random.uniform(0, 1) < eps:
                action = board.get_random_action()
            else:
                action = get_nn_move(policy_net, board, device)
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
                loss = optimise_net(
                    policy_net,
                    target_net,
                    replay_mem.sample(batch_size),
                    optimiser,
                    device)
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
# policy_net = LinearFC().to(device)
# target_net = LinearFC().to(device)
policy_net = ConvNet().to(device)
target_net = ConvNet().to(device)
# play_game(net, board)
train(
    policy_net,
    target_net,
    n_episodes=10,
    device=device,
    batch_size=64
)
