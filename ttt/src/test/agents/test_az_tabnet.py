import random

import pytest
import torch
from torch.optim import Adam

import ttt.env as t3
import agents.alphazero as az
from agents.alphazero import ResNet, GameStep
from agents.compare import play_games
from agents.random import RandomAgent
from agents.tab_greedy_v import TabGreedyVAgent
from utils.maths import softmax


@pytest.mark.slow
def test_plays_well_with_tabnet():
    batch_size = 128
    epochs = 10
    device = "cuda"
    rnet = az.ResNet(num_res_blocks=4, num_hidden=32, device=device)
    optimiser = Adam(rnet.parameters(), lr=0.001, weight_decay=0.0001)

    tab_agent = TabGreedyVAgent.load("trained_models/tmcts_sym_100k_30")
    # only train on nonterminal states, like az
    nonterminal_states = [
        (env, board, cp, value)
        for env, board, cp, value in extract_all_state_values(tab_agent)
        if env.status() == t3.IN_PROGRESS
    ]
    game_steps = [
        GameStep(
            board, cp, valid_mask(board), greedy_probs(tab_agent, env.str1d()), value
        )
        for env, board, cp, value in nonterminal_states
    ]

    results_pre = evaluate(rnet, device, n_games=50)
    policy_losses, value_losses = train(
        rnet, optimiser, game_steps, epochs, batch_size, device
    )
    results_post = evaluate(rnet, device, n_games=50)

    assert policy_losses[-1] < policy_losses[0]
    assert value_losses[-1] < value_losses[0]
    # trained agent should win more than untrained
    # NOTE this sometimes fails if u get unlucky with training
    assert results_post["X"] > results_pre["X"]


def train(
    rnet: ResNet,
    optimiser: Adam,
    game_steps: list[GameStep],
    epochs: int,
    batch_size: int,
    device: str,
):
    rnet.train()
    policy_losses, value_losses = [], []
    for _ in range(epochs):  # epochs
        for _ in range(len(game_steps) // batch_size):
            sample = random.sample(game_steps, batch_size)
            pl, vl = az._update_net(
                rnet, optimiser, sample, device=device, mask_invalid_actions=True
            )
            policy_losses.append(pl)
            value_losses.append(vl)
    return policy_losses, value_losses


def evaluate(rnet: ResNet, device: str, n_mcts_sims=10, n_games=50):
    agent = az.AlphaZeroAgent.from_nn(
        model=rnet, device=device, n_mcts_sims=n_mcts_sims, c_puct=1.0
    )
    rng = RandomAgent()
    rnet.eval()
    with torch.no_grad():
        return play_games(agent, rng, n_games=n_games)


def extract_all_state_values(tab_agent):
    q_table = tab_agent._q_table
    for env, value in q_table.values():
        yield env, env.board, env.current_player, value


def greedy_probs(tab_agent: TabGreedyVAgent, board_str: str):
    av = softmax(action_values(tab_agent, board_str))
    env = t3.TttEnv.from_str(board_str)
    if env.status() != t3.IN_PROGRESS:
        raise Exception("Don't give me terminal states! " + env.str1d_sep("|"))
    inval = set(range(9)) - set(env.valid_actions())
    for i in range(9):
        if i in inval:
            av[i] = 0
    avsum = sum(av)
    av = [a / avsum for a in av]
    return av


def action_values(tab_agent: TabGreedyVAgent, board_str: str):
    av = tab_agent.action_values(board_str)
    return [av.get(i, 0) for i in range(9)]


def valid_mask(board: t3.Board):
    valid_actions = list(t3.valid_actions(board))
    return [x in valid_actions for x in range(9)]
