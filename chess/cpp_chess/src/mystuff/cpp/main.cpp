#include <iostream>
#include <chrono>
#include "agent_random.h"
#include "chess/board.h"
#include "andoma_agent.h"
#include <vector>
#include <string>

namespace mystuff {

struct AgentInfo {
    std::string name;
    Agent* agent;
};

struct MatchResult {
    bool draw = false;
    bool white_win = false;
    int halfMoves = 0;
};

MatchResult play_game(Agent& white, Agent& black) {
    lczero::InitializeMagicBitboards();
    LeelaBoardWrapper board;
    Agent* agents[2] = {&white, &black};
    int turn = 0;
    MatchResult result;
    while (!board.is_game_over()) {
        auto move = agents[turn]->select_move(board);
        board.make_move(move);
        turn = 1 - turn;
        result.halfMoves++;
    }
    std::string res = board.result();
    if (res == "1-0") result.white_win = true;
    else if (res == "0-1") result.draw = false;
    else result.draw = true;

    return result;
}

// play all agent pairs
// todo: print time taken
void bot_fight(int n_games) {
    RandomAgent random_agent;
    AndomaAgent andoma_agent_s1(1);
    AndomaAgent andoma_agent_s2(2);
    std::vector<AgentInfo> agents = {
        {"RandomAgent", &random_agent},
        {"AndomaAgent depth=1", &andoma_agent_s1},
        {"AndomaAgent depth=2", &andoma_agent_s2}
    };
    int n = agents.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            int white_wins = 0;
            int black_wins = 0;
            int draws = 0;
            int totalHalfmoves = 0;
            auto start = std::chrono::high_resolution_clock::now();
            for (int g = 0; g < n_games; ++g) {
                auto result = play_game(*agents[i].agent, *agents[j].agent);
                totalHalfmoves += result.halfMoves;
                if (result.white_win) white_wins++;
                else if (!result.draw) black_wins++;
                else draws++;
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            double games_per_sec = n_games / elapsed.count();
            double avg_halfmoves = static_cast<double>(totalHalfmoves) / n_games;
            std::cout << agents[i].name << " vs " << agents[j].name << " : ";
            std::cout << white_wins << " wins, " << black_wins << " losses, " << draws << " draws,";
            std::cout << "  Games/sec: " << games_per_sec << ", Avg halfmoves: " << avg_halfmoves << std::endl;
        }
    }
}

} // namespace mystuff

int main() {
    mystuff::bot_fight(100);
    return 0;
}
