import math
import random
import typing as t
from ttt.agents.agent import TttAgent
import ttt.env as t3


type ValFunc = t.Callable[[t3.Env, t3.Player], float]


def random_rollout_reward(env: t3.Env, player: t3.Player):
    """ Returns the given player's reward from a random rollout """
    s = env.status()
    if s == t3.IN_PROGRESS:
        tempenv = env.copy()
        while tempenv.status() == t3.IN_PROGRESS:
            move = random.choice(list(tempenv.valid_actions()))
            tempenv.step(move)
        s = tempenv.status()
    return 1 if s == player else -1 if s == t3.other_player(player) else 0


class MctsAgent2(TttAgent):
    def __init__(self, n_sims: int, valfn: ValFunc = random_rollout_reward):
        self.n_sims = n_sims
        self._valfn = valfn

    def get_action(self, env: t3.Env):
        return _mcts_decision(env, self.n_sims, self._valfn)


def _mcts_decision(env: t3.Env, n_simulations: int, val_func: ValFunc):
    root = _build_mcts_tree(env, n_simulations, val_func)
    best_move = max(root.children, key=lambda move: root.children[move].visits)
    return best_move


class _MCTSNode:
    def __init__(self, state: t3.Env, parent):
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
        return f'v{self.visits:3} t{self.total_reward:0.2f} u{self.ucb1():0.3f}'


def _max_ucb_child(node: _MCTSNode) -> _MCTSNode:
    return max(node.children.values(), key=lambda node: node.ucb1())


def _build_mcts_tree(
        env: t3.Env,
        simulations: int,
        val_func: ValFunc
        ):
    root = _MCTSNode(env, parent=None)
    player = env.current_player  # assume we're planning for the current player
    for _ in range(simulations):
        node = root

        # select (using tree policy): trace a path to a leaf node
        while node.children:
            node = _max_ucb_child(node)

        # expand: initialise child nodes of leaf
        if node.state.status() == t3.IN_PROGRESS:
            env = node.state
            for action in env.valid_actions():
                temp_env = env.copy()
                _, reward, _, _, _ = temp_env.step(action)
                node.children[action] = _MCTSNode(temp_env, parent=node)
            # ... and pick a random node to run the simulation step
            node = random.choice(list(node.children.values()))

        # simulate/rollout. Standard MCTS does a full "rollout" here, ie. plays
        # to the end of the game. Instead, we just use the state value estimate
        reward = val_func(node.state, player)

        # propagate values back to root
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    return root
