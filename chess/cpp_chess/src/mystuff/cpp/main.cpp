#include <iostream>
#include <chrono>
#include "agent_random.h"
#include "chess/board.h"
#include "andoma_agent.h"
#include <vector>
#include <string>

namespace mystuff {

// int play_single_game() {
//     lczero::InitializeMagicBitboards();
//     RandomAgent white = RandomAgent();
//     RandomAgent black = RandomAgent();
//     Agent* agents[2] = {&white, &black};
//     LeelaBoardWrapper board = LeelaBoardWrapper();
//     int turn = 0;
//     int numHalfmoves = 0;
//     while (!board.is_game_over()) {
//         auto move = agents[turn]->select_move(board);
//         board.make_move(move);
//         turn = 1 - turn;
//         numHalfmoves++;
//     }
//     return numHalfmoves;
// }

// void play_multiple_games(int num_games) {
//     int total_moves = 0;
//     auto start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_games; ++i) {
//         total_moves += play_single_game();
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     double games_per_sec = num_games / elapsed.count();
//     double avg_game_length = static_cast<double>(total_moves) / num_games;
//     std::cout << "Played " << num_games << " games in " << elapsed.count() << " seconds.\n";
//     std::cout << "Games per second: " << games_per_sec << std::endl;
//     std::cout << "Average game length (half-moves): " << avg_game_length << std::endl;
// }

struct AgentInfo {
    std::string name;
    Agent* agent;
};

struct MatchResult {
    int white_wins = 0;
    int black_wins = 0;
    int draws = 0;
};

// Returns 1 if white wins, -1 if black wins, 0 if draw
int play_game(Agent& white, Agent& black) {
    lczero::InitializeMagicBitboards();
    LeelaBoardWrapper board;
    Agent* agents[2] = {&white, &black};
    int turn = 0;
    while (!board.is_game_over()) {
        auto move = agents[turn]->select_move(board);
        board.make_move(move);
        turn = 1 - turn;
    }
    std::string res = board.result();
    if (res == "1-0") return 1;
    if (res == "0-1") return -1;
    return 0;
}

void play_all_agents_against_each_other(int games_per_pair = 100) {
    // Add new agents here as needed
    RandomAgent random_agent;
    AndomaAgent andoma_agent;
    std::vector<AgentInfo> agents = {
        {"RandomAgent", &random_agent},
        {"AndomaAgent", &andoma_agent}
    };
    int n = agents.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            MatchResult result1, result2;
            // First match: i as white, j as black
            for (int g = 0; g < games_per_pair; ++g) {
                int res = play_game(*agents[i].agent, *agents[j].agent);
                if (res == 1) result1.white_wins++;
                else if (res == -1) result1.black_wins++;
                else result1.draws++;
            }
            // Second match: swap colors
            for (int g = 0; g < games_per_pair; ++g) {
                int res = play_game(*agents[j].agent, *agents[i].agent);
                if (res == 1) result2.white_wins++;
                else if (res == -1) result2.black_wins++;
                else result2.draws++;
            }
            std::cout << agents[i].name << " (White) vs " << agents[j].name << " (Black): ";
            std::cout << result1.white_wins << " wins, " << result1.black_wins << " losses, " << result1.draws << " draws\n";
            std::cout << agents[j].name << " (White) vs " << agents[i].name << " (Black): ";
            std::cout << result2.white_wins << " wins, " << result2.black_wins << " losses, " << result2.draws << " draws\n";
        }
    }
}

} // namespace mystuff

int main() {
    // mystuff::play_multiple_games(100);
    mystuff::play_all_agents_against_each_other(100);
    return 0;
}
