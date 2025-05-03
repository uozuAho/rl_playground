import math
import random
import typing as t
from ttt.agents.agent import TttAgent
import ttt.env as t3


# Evaluate the current env state for the given player. Expected
# to be a higher value for more favorable states for the given
# player.
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


class MctsAgent(TttAgent):
    """ Monte-carlo tree search agent.
        Good visualisation here: https://vgarciasc.github.io/mcts-viz/
        - https://github.com/vgarciasc/mcts-viz/
        - my fork: https://github.com/uozuAho/mcts-viz
    """
    def __init__(
            self,
            n_sims: int,
            valfn: ValFunc = random_rollout_reward,
            use_valfn_for_expand = False):
        self.n_sims = n_sims
        self._valfn = valfn
        self._use_valfn_for_expand = use_valfn_for_expand

    def get_action(self, env: t3.Env):
        return _mcts_decision(env, self.n_sims, self._valfn, self._use_valfn_for_expand)

    def print_tree(self, env: t3.Env, n_sims=-1):
        """ For debugging """
        n_sims = n_sims if n_sims > 0 else self.n_sims
        tree = _build_mcts_tree(env, n_sims, self._valfn, self._use_valfn_for_expand)
        print_tree(tree)


def _mcts_decision(
        env: t3.Env,
        n_simulations: int,
        val_func: ValFunc,
        use_val_func_for_expand: bool
        ):
    root = _build_mcts_tree(env, n_simulations, val_func, use_val_func_for_expand)
    best_move = max(root.children, key=lambda move: (
        root.children[move].visits, root.children[move].total_reward))
    return best_move


class _MCTSNode:
    def __init__(self, state: t3.Env, parent):
        self.state = state
        self.parent: _MCTSNode = parent
        self.children: t.Dict[int, _MCTSNode] = {}  # action, node
        self.visits = 0
        self.total_reward = 0.0  # sum of all rewards/estimates from all visited children

    def who_moved_last(self):
        return t3.other_player(self.state.current_player)

    def ucb1(self):
        if not self.parent:
            return float('NaN')
        if self.visits == 0:
            return float('inf')
        return self.total_reward / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)

    def __str__(self):
        return f'{self.state.str1d()} vis{self.visits:3} tval{self.total_reward:5.2f} uct{self.ucb1():5.2f}'

    def __repr__(self):
        pv = self.parent.visits if self.parent else 0
        return f'v{self.visits:3} pv{pv:3} t{self.total_reward:5.2f} u{self.ucb1():5.2f}'


def print_tree(root: _MCTSNode, action=-1, indent=0):
    print(f'{" "*indent}{action}: {root}')
    for action, node in root.children.items():
        print_tree(node, action, indent + 4)


def _build_mcts_tree(
        env: t3.Env,
        simulations: int,
        val_func: ValFunc,
        use_val_func_for_expand
        ):
    root = _MCTSNode(env, parent=None)
    for _ in range(simulations):
        node = root

        # select (using tree policy): trace a path to a leaf node
        while node.children:
            maxucb = -9999999
            for c in node.children.values():
                ucb = c.ucb1()
                if ucb > maxucb:
                    maxucb = ucb
                    maxchild = c
            node = maxchild

        # expand: initialise child nodes of leaf
        if node.state.status() == t3.IN_PROGRESS:
            for action in node.state.valid_actions():
                child_state = node.state.copy()
                child_state.step(action)
                node.children[action] = _MCTSNode(child_state, parent=node)
            if use_val_func_for_expand:
                node = max(node.children.values(),
                           key=lambda c: val_func(c.state, node.state.current_player))
            else:
                node = random.choice(list(node.children.values()))

        # simulate/rollout. Standard MCTS does a full "rollout" here, ie. plays
        # to the end of the game. Instead, we just use the state value estimate
        # todo: this should use the real reward for terminal states
        rewarded_player = node.who_moved_last()
        reward = val_func(node.state, node.who_moved_last())

        # propagate values back to root
        while node:
            node.visits += 1
            if node.who_moved_last() == rewarded_player:
                node.total_reward += reward
            else:
                node.total_reward -= reward
            node = node.parent

    return root
