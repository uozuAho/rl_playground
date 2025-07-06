#pragma once
#include "agent.h"
#include <string>

namespace mystuff {

// A C++ implementation of the AndomaAgent (minimax, Michniewski evaluation)
class AndomaAgent : public Agent {
public:
    AndomaAgent(int search_depth = 1);
    lczero::Move select_move(const LeelaBoardWrapper& board) override;
    std::string name() const override { return "AndomaAgent"; }
private:
    int search_depth_;
};

} // namespace mystuff
