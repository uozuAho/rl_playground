""" Attempt to demonstrate MCTS with tic tac toe

Todo
- configurable agents: mcts vs random/mcts/perfect. Show win rates
"""


import math
import random
import typing as t

import ttt.env
from ttt.env import TicTacToeEnv


type Player = t.Literal['O', 'X']
type GameState = t.Literal['O', 'X', 'draw', 'in_progress']


def gamestate(env: TicTacToeEnv) -> GameState:
    state = env.get_status()
    if state == ttt.env.O_WIN: return 'O'
    if state == ttt.env.X_WIN: return 'X'
    if state == ttt.env.DRAW: return 'draw'
    return 'in_progress'


class MCTSNode:
    def __init__(self, state: TicTacToeEnv, parent):
        self.state: TicTacToeEnv = state
        self.parent: MCTSNode = parent
        self.children: t.Dict[int, MCTSNode] = {}  # action, node
        self.visits = 0
        self.wins = 0

    def __str__(self):
        board_str = ''.join(ttt.env.tomark(x) for x in self.state.board)
        board_str = f'{board_str[:3]}|{board_str[3:6]}|{board_str[6:]}'
        return f'vis: {self.visits:3} wins: {self.wins:3} ucb {self.ucb1():0.3f} board: {board_str}'

    def __repr__(self):
        return str(self)

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
        while gamestate(sim_state) == 'in_progress':
            # rollout policy: both players make random moves
            move = random.choice(list(sim_state.valid_actions()))
            sim_state.step(move)
        return gamestate(sim_state)

    def backpropagate(self, player: Player, result: GameState):
        assert result != 'in_progress'
        if result == player:
            self.wins += 1
        elif result != 'draw':
            self.wins -= 1
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(player, result)

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)


def build_mcts_tree(state: TicTacToeEnv, simulations: int):
    root = MCTSNode(state, parent=None)
    for _ in range(simulations):
        node = root
        while node.children:
            node = node.select()    # 1. select until a leaf is found

        if gamestate(node.state) == 'in_progress':
            node.expand()           # 2. generate & choose a child of the leaf
            if node.children:
                node = random.choice(list(node.children.values()))

        result = node.simulate()    # 3. simulate from the given child to the
                                    #    end of the game

        # 4. propagate game result to all nodes that led to the result
        node.backpropagate(state.current_player, result)
    return root


def mcts_decision(state: TicTacToeEnv, simulations=1000):
    root = build_mcts_tree(state, simulations)
    best_move = max(root.children, key=lambda move: root.children[move].visits)
    return best_move


game = TicTacToeEnv()
done = False

while not done:
    if game.current_player == 'X':
        move = mcts_decision(game, simulations=100)
    else:
        move = random.choice(list(game.valid_actions()))
    _, _, done, _, _ = game.step(move)
    print(f"Player {game.current_player} moved to {move}")
    game.render()
    user_done = False

print(f"Winner: {gamestate(game)}")
