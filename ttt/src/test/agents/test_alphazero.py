from torch.optim import Adam

from agents import alphazero
from agents.random import RandomAgent
import algs.az_evaluators as az_eval
import ttt.env as t3


def test_train_and_play_random():
    az = alphazero.train_new(1, "cpu")

    rng = RandomAgent()
    env = t3.TttEnv()
    done = False
    while not done:
        if env.current_player == t3.X:
            action = az.get_action(env)
        else:
            action = rng.get_action(env)
        _, _, done, _, _ = env.step(action)


def test_train_and_play_parallel():
    model = alphazero.ResNet(1, 1, "cpu")
    optimiser = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    alphazero.train(
        model,
        optimiser,
        n_games=5,
        n_epochs=2,
        n_mcts_sims=5,
        c_puct=2.0,
        temperature=1.25,
        device="cpu",
        mask_invalid_actions=False,
        train_batch_size=4,
        parallel=True,
    )
    az = alphazero.AlphaZeroAgent.from_nn(
        model, device="cpu", n_mcts_sims=10, c_puct=1.0
    )

    rng = RandomAgent()
    env = t3.TttEnv()
    done = False
    while not done:
        if env.current_player == t3.X:
            action = az.get_action(env)
        else:
            action = rng.get_action(env)
        _, _, done, _, _ = env.step(action)


def test_plays_with_dummy_eval():
    az = alphazero.AlphaZeroAgent.from_eval(mcts_eval=az_eval.uniform, n_mcts_sims=10)

    rng = RandomAgent()
    env = t3.TttEnv()
    done = False
    while not done:
        if env.current_player == t3.X:
            action = az.get_action(env)
        else:
            action = rng.get_action(env)
        _, _, done, _, _ = env.step(action)


def test_x_should_block_win():
    env = t3.TttEnv.from_str("..x|.oo|..x")
    agent = alphazero.AlphaZeroAgent.from_eval(
        n_mcts_sims=100, mcts_eval=az_eval.uniform, c_puct=10.0
    )
    action = agent.get_action(env)
    assert action == 3


def test_o_should_block_win():
    env = t3.TttEnv.from_str("x.o|.xx|..o")
    agent = alphazero.AlphaZeroAgent.from_eval(
        n_mcts_sims=100, mcts_eval=az_eval.uniform
    )
    action = agent.get_action(env)
    assert action == 3


def test_x_should_win():
    env = t3.TttEnv.from_str("x.o|.xx|.oo")
    agent = alphazero.AlphaZeroAgent.from_eval(
        n_mcts_sims=30, mcts_eval=az_eval.uniform
    )
    action = agent.get_action(env)
    assert action == 3


def test_o_should_win():
    env = t3.TttEnv.from_str("oxx|.oo|.xx")
    agent = alphazero.AlphaZeroAgent.from_eval(
        n_mcts_sims=30, mcts_eval=az_eval.uniform
    )
    action = agent.get_action(env)
    assert action == 3
