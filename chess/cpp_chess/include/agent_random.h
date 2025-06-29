#pragma once
#include <vector>
#include <random>
#include "agent.h"

// Forward declaration for LeelaChessZero board
namespace lcz {
    class Board;
    struct Move;
}

class RandomAgent : public Agent {
public:
    RandomAgent();
    lcz::Move select_move(const lcz::Board& board) override;
    std::string name() const override { return "RandomAgent"; }
private:
    std::mt19937 rng;
};
