import typing as t

import chess
from collections import deque
from dataclasses import dataclass
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from lib.env import CaptureChess
from lib.nets import ChessNet


type EpCallback = t.Callable[[int], None]  # func(ep_number)


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


def epsilon(eps_start, eps_end, n_total_ep, ep):
    return eps_end + (eps_start - eps_end) * math.exp(-6.0 * ep / n_total_ep)


def update_target(policy_net: ChessNet, target_net: ChessNet, tau=1.0):
    """(Soft) Update the target network. Tau = hardness. If Tau = 1.0, it's a
    hard update, ie the target net is set to equal the policy net.
    """
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * tau + target_net_state_dict[key] * (1 - tau)
    target_net.load_state_dict(target_net_state_dict)


def is_endstate(layer_board: np.ndarray):
    return np.array_equal(layer_board, layer_board * 0)


def optimise_net(
    policy_net: ChessNet,
    target_net: ChessNet,
    transitions: list[Transition],
    optimiser: optim.Optimizer,
    device: str,
    gamma=0.99,
):
    """Update NN weights using a batch of transitions. Returns loss"""

    batch_size = len(transitions)
    states = torch.stack([torch.from_numpy(t.state) for t in transitions]).to(device)
    moves_list = [(t.move[0] * 64 + t.move[1]) for t in transitions]
    moves = torch.tensor(moves_list, dtype=torch.long, device=device)
    rewards = torch.tensor(
        [t.reward for t in transitions], dtype=torch.float64, device=device
    )
    next_states = torch.stack([torch.from_numpy(t.next_state) for t in transitions]).to(
        device
    )

    qs = policy_net(states)
    qsa = qs.gather(1, moves.unsqueeze(1)).squeeze(1)

    non_final_mask_list = [not is_endstate(t.next_state) for t in transitions]
    non_final_mask = torch.tensor(non_final_mask_list, dtype=torch.bool, device=device)
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


def get_nn_move(net: nn.Module, board: CaptureChess, device) -> chess.Move:
    """Assumes a net with a 1x4096 (64x64) output, which represents a
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
    moves = [
        x
        for x in board.board.generate_legal_moves()
        if x.from_square == move_from and x.to_square == move_to
    ]
    if len(moves) == 0:
        # If all legal moves have negative action value, explore
        move = board.get_random_action()
        move_from = move.from_square
        move_to = move.to_square
    else:
        # If there are multiple max-moves, pick a random one
        move = random.choice(moves)
    return move


def train(
    policy_net: ChessNet,
    target_net: ChessNet,
    n_episodes: int,
    device: str,
    n_episode_action_limit=25,
    batch_size=32,
    target_net_update_eps=10,
    target_net_update_tau=1.0,
    ep_callback: t.Optional[EpCallback] = None,
):
    """Returns [losses], [rewards]. One value per episode."""
    board = CaptureChess(action_limit=n_episode_action_limit)
    optimiser = optim.SGD(policy_net.parameters(), lr=1e-4)
    episode = 0
    ep_losses = []
    ep_rewards = []
    replay_mem = ReplayMemory(1000)
    for ep in range(n_episodes):
        try:
            print(f"{ep}/{n_episodes}")
            if ep % target_net_update_eps == 0:
                update_target(policy_net, target_net, target_net_update_tau)
            losses: list[float] = []
            rewards: list[float] = []
            board.reset()
            game_over = False
            eps = epsilon(0.99, 0.01, n_episodes, ep)
            while not game_over and len(rewards) < n_episode_action_limit:
                state = board.layer_board
                if np.random.uniform(0, 1) < eps:
                    action = board.get_random_action()
                else:
                    action = get_nn_move(policy_net, board, device)
                game_over, reward = board.step(action)
                next_state = board.layer_board
                rewards.append(reward)
                episode += 1

                replay_mem.push(
                    Transition(
                        state,
                        (action.from_square, action.to_square),
                        next_state,
                        reward,
                    )
                )

                if len(replay_mem) >= batch_size:
                    loss = optimise_net(
                        policy_net,
                        target_net,
                        replay_mem.sample(batch_size),
                        optimiser,
                        device,
                    )
                    losses.append(loss)
            ep_losses.append(sum(losses))
            ep_rewards.append(sum(rewards))

            if ep_callback:
                ep_callback(ep)
        except KeyboardInterrupt:
            break
    return ep_losses, ep_rewards
