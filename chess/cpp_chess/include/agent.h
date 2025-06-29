#pragma once
#include <vector>
#include <string>

// Forward declaration for LeelaChessZero board
namespace lcz {
    class Board;
    struct Move;
}

class Agent {
public:
    virtual ~Agent() = default;
    // Returns a move in UCI or internal format, as appropriate
    virtual lcz::Move select_move(const lcz::Board& board) = 0;
    virtual std::string name() const = 0;
};
