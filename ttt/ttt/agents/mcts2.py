from dataclasses import dataclass
import math
import random
import typing as t
from ttt.agents.agent import TttAgent
import ttt.env as t3


type ValFunc = t.Callable[[t3.Env], float]


def random_rollout_reward(env: t3.Env):
    """ Returns the reward from a random rollout. Reward is 1 for a win for
        the current player.
    """
    player = env.current_player
    s = env.status()
    if s == t3.IN_PROGRESS:
        tempenv = env.copy()
        while tempenv.status() == t3.IN_PROGRESS:
            move = random.choice(list(tempenv.valid_actions()))
            tempenv.step(move)
        s = tempenv.status()
    return 1 if s == player else -1 if s == t3.other_player(player) else 0


def random_rollout_win(env: t3.Env):
    """ Returns 1 if random rollout results in a win for the current player.
    """
    player = env.current_player
    s = env.status()
    if s == t3.IN_PROGRESS:
        tempenv = env.copy()
        while tempenv.status() == t3.IN_PROGRESS:
            move = random.choice(list(tempenv.valid_actions()))
            tempenv.step(move)
        s = tempenv.status()
    return 1 if s == player else 0


class MctsAgent2(TttAgent):
    def __init__(self, n_sims: int, valfn: ValFunc = random_rollout_win):
        self.n_sims = n_sims
        self._valfn = valfn

    def get_action(self, env: t3.Env):
        return _mcts_decision(env, self.n_sims, self._valfn)


def _mcts_decision(env: t3.Env, n_simulations: int, val_func: ValFunc):
    root = _build_mcts_tree(env, n_simulations, val_func)
    best_move = max(root.children, key=lambda move: root.children[move].visits)
    return best_move


@dataclass
class GameState:
    env: t3.Env
    is_terminal: bool
    reward: int


class _MCTSNode:
    def __init__(self, state: GameState, parent):
        self.state = state
        self.parent: _MCTSNode = parent
        self.children: t.Dict[int, _MCTSNode] = {}  # action, node
        self.visits = 0
        self.total_reward = 0.0  # sum of all rewards/estimates from all visited children

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return self.total_reward / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)

    def __repr__(self):
        end = ''
        if self.state.is_terminal:
            end = 'WIN' if self.state.reward > 0 else 'LOSS'
        return f'v{self.visits} t{self.total_reward:.2f}  u{self.ucb1():.2f} {end}'


def _max_ucb_child(node: _MCTSNode) -> _MCTSNode:
    return max(node.children.values(), key=lambda node: node.ucb1())


def _build_mcts_tree(
        env: t3.Env,
        simulations: int,
        val_func: ValFunc
        ):
    root = _MCTSNode(GameState(env, False, 0), parent=None)
    for _ in range(simulations):
        node = root

        # select (using tree policy): trace a path to a leaf node
        while node.children:
            node = _max_ucb_child(node)

        # expand: initialise child nodes of leaf
        if node.state.env.status() == t3.IN_PROGRESS:
            env = node.state.env
            for action in env.valid_actions():
                temp_env = env.copy()
                _, reward, term, trunc, _ = temp_env.step(action)
                done = term or trunc
                node.children[action] = _MCTSNode(
                    GameState(temp_env, done, reward), parent=node)
            # ... and pick a random node to run the simulation step
            node = random.choice(list(node.children.values()))

        # simulate/rollout. Standard MCTS does a full "rollout" here, ie. plays
        # to the end of the game. Instead, we just use the state value estimate
        reward = val_func(node.state.env)

        # propagate values back to root
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    return root
