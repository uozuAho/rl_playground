#pragma once
#include <string>

namespace mystuff {

struct Agent;
struct AgentInfo {
    std::string name;
    Agent* agent;
};

struct MatchResult {
    bool draw = false;
    bool white_win = false;
    int halfMoves = 0;
};

MatchResult play_game(Agent& white, Agent& black);
void bot_fight(int n_games);

} // namespace mystuff
