#pragma once
#include <vector>
#include <string>

// Forward declaration for LeelaChessZero board
namespace lczero {
    class Board;
    struct Move;
}

class Agent {
public:
    virtual ~Agent() = default;
    // Returns a move in UCI or internal format, as appropriate
    virtual lczero::Move select_move(const lczero::Board& board) = 0;
    virtual std::string name() const = 0;
};
