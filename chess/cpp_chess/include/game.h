#pragma once
#include <memory>
#include "agent.h"
#include "leela_board_wrapper.h"

class Game {
public:
    Game(Agent* white, Agent* black);
    void play();
    std::string result() const;
private:
    Agent* white_;
    Agent* black_;
    LeelaBoardWrapper board_;
    std::vector<std::string> move_history_;
};
