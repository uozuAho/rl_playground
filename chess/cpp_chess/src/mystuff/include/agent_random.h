#pragma once
#include <vector>
#include <random>
#include "agent.h"

// Forward declaration for LeelaChessZero board
namespace lczero {
    class Board;
    struct Move;
}

class RandomAgent : public Agent {
public:
    RandomAgent();
    lczero::Move select_move(const lczero::Board& board) override;
    std::string name() const override { return "RandomAgent"; }
private:
    std::mt19937 rng;
};
