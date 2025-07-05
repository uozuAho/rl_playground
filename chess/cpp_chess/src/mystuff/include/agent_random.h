#pragma once
#include <vector>
#include <random>
#include "agent.h"

class RandomAgent : public Agent {
public:
    RandomAgent();
    lczero::Move select_move(const lczero::ChessBoard& board) override;
    std::string name() const override { return "RandomAgent"; }
private:
    std::mt19937 rng;
};
