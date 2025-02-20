from RLC.move_chess.environment import Board
from RLC.move_chess.agent import Piece
from RLC.move_chess.learn import Reinforce

# -----------------------------------------------
# 1. policy iteration
env = Board()
env.render()
# for line in env.visual_board:
#     print(line)

p = Piece(piece='king')
r = Reinforce(p,env)
# print(r.agent.value_function.astype(int))
# r.evaluate_policy(gamma=1)
# print(r.agent.value_function.astype(int))


# -----------------------------------------------
# 2. model free methods (no game model)
# for k in range(100):
#     eps = 0.5
#     r.monte_carlo_learning(epsilon=eps)
# r.visualize_policy()
