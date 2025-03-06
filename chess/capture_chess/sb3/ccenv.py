import chess
import gymnasium as gym
from gymnasium import spaces
from RLC.capture_chess.environment import Board


ILLEGAL_ACTION_THROW = 0
ILLEGAL_ACTION_NEG_REWARD = 1
ILLEGAL_ACTION_NEG_REWARD_GAME_OVER = 2


class IllegalActionError(Exception):
    def __init__(self, board: Board, move: chess.Move):
        super().__init__()
        self.board = board
        self.move = move


class CaptureChess(gym.Env):
    """ Gym wrapper around capture chess environment """
    def __init__(
            self,
            illegal_action_behaviour,
            n_action_limit=25,
            allow_pawn_promotion=False):
        self._board = Board()
        self.action_space = spaces.Discrete(4096)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,8,8))
        self.illegal_action_behaviour = illegal_action_behaviour
        self.n_action_limit = n_action_limit
        self.allow_pawn_promotion = allow_pawn_promotion
        self.n_actions = 0

    def reset(self, **kwargs):
        self._board.reset()
        self.n_actions = 0
        return self._board.layer_board, {}

    def step(self, action: int):
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)
        if not self._board.board.is_legal(move):
            if self.illegal_action_behaviour == ILLEGAL_ACTION_THROW:
                raise IllegalActionError(self._board, move)
            if self.illegal_action_behaviour == ILLEGAL_ACTION_NEG_REWARD:
                return self._board.layer_board, -100, False, False, {}
            if self.illegal_action_behaviour == ILLEGAL_ACTION_NEG_REWARD_GAME_OVER:
                return self._board.layer_board, -100, True, False, {}
        self.n_actions += 1
        truncated = False
        done, reward = self._board.step(move)
        # hack fix pawn promotion reward
        # reward should only be 1,3,5,9
        if reward % 2 == 0:
            reward = 0
        # hack: if there are no legal actions, the game is over
        # This fixes when only pawn promotions are legal
        if len(list(self.legal_actions())) == 0:
            done = True
        if self.n_actions >= self.n_action_limit:
            truncated = True
            reward = 0
        return self._board.layer_board, reward, done, truncated, { "value": self._board.get_material_value()}

    def render(self):
        print(self._board.board)

    def legal_actions(self):
        for move in self._board.board.generate_legal_moves():
            # 64x64 output doesn't support pawn promotions, just remove them
            if self.allow_pawn_promotion or not move.promotion:
                yield move.from_square * 64 + move.to_square

    def current_obs(self):
        return self._board.layer_board

    def action_masks(self):
        mask = self._board.project_legal_moves().reshape(4096)
        # 64x64 output doesn't support pawn promotions, just remove them
        for m in self._board.board.generate_legal_moves():
            if not self.allow_pawn_promotion and m.promotion:
                mask[m.from_square * 64 + m.to_square] = 0
        return mask
