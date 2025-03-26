""" Same as greedy v, but use MCTS for planning """

from dataclasses import dataclass
import math
import random
import typing as t
import torch
from ttt.agents.torch_greedy_v import GreedyTdAgent
import ttt.env2 as ttt


type ValFunc = t.Callable[[ttt.Board], float]


@dataclass
class GameState:
    env: ttt.Env
    is_terminal: bool
    reward: int


class MCTSNode:
    def __init__(self, state: GameState, parent):
        self.state = state
        self.parent: MCTSNode = parent
        self.children: t.Dict[int, MCTSNode] = {}  # action, node
        self.visits = 0
        self.total_reward = 0.0  # sum of all rewards/estimates from all visited children

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return self.total_reward / self.visits + math.sqrt(math.log(self.parent.visits) / self.visits)

    def __repr__(self):
        b = ''.join('x' if c == ttt.X else 'o' if c == ttt.O else '.' for c in self.state.env.board)
        end = ''
        if self.state.is_terminal:
            end = 'WIN' if self.state.reward > 0 else 'LOSS'
        return f'{b[:3]}|{b[3:6]}|{b[6:]} : t{self.total_reward:.2f} v{self.visits} u{self.ucb1():.2f} {end}'


def max_ucb_child(node: MCTSNode) -> MCTSNode:
    return max(node.children.values(), key=lambda node: node.ucb1())


def build_mcts_tree(
        env: ttt.Env,
        simulations: int,
        val_func: ValFunc
        ):
    root = MCTSNode(GameState(env, False, 0), parent=None)
    for _ in range(simulations):
        node = root

        # select (using tree policy): trace a path to a leaf node
        # todo: make sure this explores
        while node.children:
            node = max_ucb_child(node)

        # expand: initialise child nodes of leaf
        if not node.state.is_terminal:
            env = node.state.env
            for action in env.valid_actions():
                temp_env = env.copy()
                _, reward, term, trunc, _ = temp_env.step(action)
                done = term or trunc
                node.children[action] = MCTSNode(
                    GameState(temp_env, done, reward), parent=node)
        # ... and pick a random node to run the simulation step
        node = random.choice(list(node.children.values()))

        # simulate/rollout. Standard MCTS does a full "rollout" here, ie. plays
        # to the end of the game. Instead, we just use the state value estimate
        if node.state.is_terminal:
            reward = node.state.reward
        else:
            reward = val_func(node.state.env.board)

        # propagate values back to root
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    return root


def mcts_decision(env: ttt.Env, n_simulations: int, val_func: ValFunc):
    root = build_mcts_tree(env, n_simulations, val_func)
    best_move = max(root.children, key=lambda move: root.children[move].visits)
    return best_move


device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = 'cpu'
print(f'using device {device}')

path = "conv-greedy-v.pth"
agent = GreedyTdAgent.load(path, device)
env = ttt.Env()
mcts_decision(env, 1000, agent.board_val)
