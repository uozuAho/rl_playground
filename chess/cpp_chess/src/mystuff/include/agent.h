#pragma once
#include <vector>
#include <string>

#include "chess/types.h"
#include "chess/board.h"

class Agent {
public:
    virtual ~Agent() = default;
    // Returns a move in UCI or internal format, as appropriate
    virtual lczero::Move select_move(const lczero::ChessBoard& board) = 0;
    virtual std::string name() const = 0;
};
