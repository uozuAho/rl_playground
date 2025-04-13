""" Same as greedy v, but use MCTS for planning. No training - uses the value
    model learned by the greedy agent.

    Slow, but wins ~97% as x against a random opponent with 20 sims.

    Todo:
    - play as o. Should be able to
"""

from dataclasses import dataclass
import math
import random
import typing as t
from ttt.agents.agent import TttAgent
from ttt.agents.torch_nn_greedy_v import NnGreedyVAgent
import ttt.env as t3


type ValFunc = t.Callable[[t3.Board], float]


class NnGreedyVMctsAgent(TttAgent):
    def __init__(self, trained_agent: NnGreedyVAgent, n_simulations: int):
        self.agent = trained_agent
        self.n_simulations = n_simulations

    def get_action(self, env: t3.Env) -> int:
        return mcts_decision(env, self.n_simulations, self._val_func)

    @staticmethod
    def load(name_or_path, n_simulations: int):
        agent = NnGreedyVAgent.load(name_or_path, device='cuda')
        return NnGreedyVMctsAgent(agent, n_simulations)

    def _val_func(self, board: t3.Board):
        return self.agent.board_val(board)


def mcts_decision(env: t3.Env, n_simulations: int, val_func: ValFunc):
    root = build_mcts_tree(env, n_simulations, val_func)
    best_move = max(root.children, key=lambda move: root.children[move].visits)
    return best_move


@dataclass
class GameState:
    env: t3.Env
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
        b = ''.join('x' if c == t3.X else 'o' if c == t3.O else '.' for c in self.state.env.board)
        end = ''
        if self.state.is_terminal:
            end = 'WIN' if self.state.reward > 0 else 'LOSS'
        return f'{b[:3]}|{b[3:6]}|{b[6:]} : t{self.total_reward:.2f} v{self.visits} u{self.ucb1():.2f} {end}'


def max_ucb_child(node: MCTSNode) -> MCTSNode:
    return max(node.children.values(), key=lambda node: node.ucb1())


def build_mcts_tree(
        env: t3.Env,
        simulations: int,
        val_func: ValFunc
        ):
    root = MCTSNode(GameState(env, False, 0), parent=None)
    for _ in range(simulations):
        node = root

        # select (using tree policy): trace a path to a leaf node
        # todo: make sure this explores
        # todo: make sure this picks known wins
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
