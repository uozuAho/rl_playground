from math import log
import random
import typing as t

import chess
import lib.env as env
from lib.agent import ChessAgent


# Evaluate the current env state for the given player. Expected
# to be a higher value for more favorable states for the given
# player.
type ValFunc = t.Callable[[env.ChessGame, env.Player], float]


def random_rollout_reward(env: env.ChessGame, player: env.Player, max_depth=50):
    """ Returns the given player's reward from a random rollout """
    winner = env.winner()
    if not env.is_game_over():
        done = False
        tempenv = env.copy()
        i = 0
        while not done:
            i += 1
            if i >= max_depth:
                break
            move = random.choice(list(tempenv.legal_moves()))
            done, _ = tempenv.step(move)
        winner = tempenv.winner()
    return 0 if not winner else 1 if winner == player else -1


class MctsAgent(ChessAgent):
    """ Monte-carlo tree search agent.
        Good visualisation here: https://vgarciasc.github.io/mcts-viz/
        - https://github.com/vgarciasc/mcts-viz/
        - my fork: https://github.com/uozuAho/mcts-viz
    """
    def __init__(
            self,
            player: env.Player,
            n_sims: int,
            valfn: ValFunc = random_rollout_reward,
            use_valfn_for_expand = False):
        self.player = player
        self.n_sims = n_sims
        self._valfn = valfn
        self._use_valfn_for_expand = use_valfn_for_expand

    def get_action(self, env: env.ChessGame):
        assert env.turn == self.player
        return _mcts_decision(env, self.n_sims, self._valfn, self._use_valfn_for_expand)

    def print_tree(self, env: env.ChessGame, n_sims=-1):
        """ For debugging """
        n_sims = n_sims if n_sims > 0 else self.n_sims
        tree = _build_mcts_tree(env, n_sims, self._valfn, self._use_valfn_for_expand)
        print_tree(tree, None)


def _mcts_decision(
        env: env.ChessGame,
        n_simulations: int,
        val_func: ValFunc,
        use_val_func_for_expand: bool
        ):
    root = _build_mcts_tree(env, n_simulations, val_func, use_val_func_for_expand)
    best_move = max(root.children, key=lambda move: (
        root.children[move].visits, root.children[move].total_reward))
    return best_move


class _MCTSNode:
    def __init__(self, state: env.ChessGame, parent, val_est: float | None = None):
        self.state = state
        self.parent: _MCTSNode = parent
        self.val_est = val_est
        self.children: t.Dict[chess.Move, _MCTSNode] = {}  # action, node
        self.visits = 0
        self.total_reward = 0.0  # sum of all rewards/estimates from all visited children

    def who_moved_last(self):
        return env.other_player(self.state.turn)

    def ucb1(self):
        v = self.visits
        p = self.parent
        return (
            float('nan') if p is None else
            float('inf') if v == 0 else
            self.total_reward / v + (2 * log(p.visits) / v) ** 0.5
        )

    def __str__(self):
        return f'{self.state} vis{self.visits:3} tval{self.total_reward:5.2f} uct{self.ucb1():5.2f}'

    def __repr__(self):
        pv = self.parent.visits if self.parent else 0
        return f'v{self.visits:3} pv{pv:3} t{self.total_reward:5.2f} u{self.ucb1():5.2f}'


def print_tree(root: _MCTSNode, action: chess.Move | None, indent=0):
    print(f'{" "*indent}{action}: {root}')
    for action, node in root.children.items():
        print_tree(node, action, indent + 4)


def _build_mcts_tree(
        env: env.ChessGame,
        simulations: int,
        val_func: ValFunc,
        use_val_func_for_expand
        ):
    root = _MCTSNode(env, parent=None)
    fen_start = env.fen()
    for _ in range(simulations):
        node = root

        # select (using tree policy): trace a path to a leaf node
        while node.children:
            maxucb = -9999999
            maxchild = node
            for c in node.children.values():
                ucb = c.ucb1()
                if ucb > maxucb:
                    maxucb = ucb
                    maxchild = c
            node = maxchild

        # expand: initialise child nodes of leaf
        if not node.state.is_game_over():
            for action in node.state.legal_moves():
                child_state = node.state.copy()
                child_state.step(action)
                if use_val_func_for_expand:
                    val = val_func(child_state, node.state.turn)
                else:
                    val = None
                node.children[action] = _MCTSNode(child_state, parent=node, val_est=val)
            if use_val_func_for_expand:
                maxval = -99999.0
                maxchild = node
                for c in node.children.values():
                    if c.val_est > maxval:  # type: ignore
                        maxval = c.val_est  # type: ignore
                        maxchild = c
                node = maxchild
            else:
                node = random.choice(list(node.children.values()))

        # simulate/rollout. Standard MCTS does a full "rollout" here, ie. plays
        # to the end of the game. Instead, we just use the state value estimate
        # todo: this should use the real reward for terminal states
        rewarded_player = node.who_moved_last()
        reward = node.val_est or val_func(node.state, node.who_moved_last())

        # propagate values back to root
        while node:
            node.visits += 1
            if node.who_moved_last() == rewarded_player:
                node.total_reward += reward
            else:
                node.total_reward -= reward
            node = node.parent

    assert root.state.fen() == fen_start

    return root
