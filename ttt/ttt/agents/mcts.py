import math
import random
import typing as t
from ttt.agents.agent import TttAgent
import ttt.env as t3


class MctsAgent(TttAgent):
    def __init__(self, n_sims: int):
        self.n_sims = n_sims

    def get_action(self, env: t3.Env):
        return mcts_decision(env, self.n_sims)


class MCTSNode:
    def __init__(self, state: t3.Env, parent):
        self.state: t3.Env = state
        self.parent: MCTSNode = parent
        self.children: t.Dict[int, MCTSNode] = {}  # action, node
        self.visits = 0
        self.wins = 0

    def __str__(self):
        board_str = ''.join(t3.tomark(x) for x in self.state.board)
        board_str = f'{board_str[:3]}|{board_str[3:6]}|{board_str[6:]}'
        return f'vis: {self.visits:3} wins: {self.wins:3} ucb {self.ucb1():0.3f} board: {board_str}'

    def __repr__(self):
        return f'v:{self.visits:3} w:{self.wins:3} ucb:{self.ucb1():0.3f}'

    def select(self):
        # tree policy: greedy best
        return max(self.children.values(), key=lambda node: node.ucb1())

    def expand(self):
        moves = self.state.valid_actions()
        for move in moves:
            new_state = self.state.copy()
            new_state.step(move)
            self.children[move] = MCTSNode(new_state, self)

    def simulate(self):
        sim_state = self.state.copy()
        while sim_state.status() == t3.IN_PROGRESS:
            # rollout policy: both players make random moves
            move = random.choice(list(sim_state.valid_actions()))
            sim_state.step(move)
        return sim_state.status()

    def backpropagate(self, player: t3.Player, result: t3.Status):
        assert result != t3.IN_PROGRESS
        if result == player:
            self.wins += 1
        elif result != t3.DRAW:
            self.wins -= 1
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(player, result)

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)


def build_mcts_tree(state: t3.Env, simulations: int):
    root = MCTSNode(state, parent=None)
    for _ in range(simulations):
        node = root
        while node.children:
            node = node.select()    # 1. select until a leaf is found

        if node.state.status() == t3.IN_PROGRESS:
            node.expand()           # 2. generate & choose a child of the leaf
            if node.children:
                node = random.choice(list(node.children.values()))

        result = node.simulate()    # 3. simulate from the given child to the
                                    #    end of the game

        # 4. propagate game result to all nodes that led to the result
        node.backpropagate(state.current_player, result)
    return root


def mcts_decision(state: t3.Env, n_simulations: int):
    root = build_mcts_tree(state, n_simulations)
    best_move = max(root.children, key=lambda move: root.children[move].visits)
    return best_move
