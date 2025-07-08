#pragma once
#include "leela_board_wrapper.h"

namespace mystuff {

// Evaluate the board position. Returns positive for white advantage, negative for black.
int evaluate_board(const LeelaBoardWrapper& board);

// Evaluate a move (stub for now)
float move_value(const LeelaBoardWrapper& board, int move, bool endgame);

} // namespace mystuff
