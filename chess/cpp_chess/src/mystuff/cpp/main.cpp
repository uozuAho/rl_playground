#include <iostream>
#include <chrono>
#include "agent_random.h"
#include "chess/board.h"

namespace mystuff {

int play_single_game() {
    lczero::InitializeMagicBitboards();
    RandomAgent white = RandomAgent();
    RandomAgent black = RandomAgent();
    Agent* agents[2] = {&white, &black};
    LeelaBoardWrapper board = LeelaBoardWrapper();
    int turn = 0;
    int numHalfmoves = 0;
    while (!board.is_game_over()) {
        auto move = agents[turn]->select_move(board);
        board.make_move(move);
        turn = 1 - turn;
        numHalfmoves++;
    }
    return numHalfmoves;
}

void play_multiple_games(int num_games) {
    int total_moves = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_games; ++i) {
        total_moves += play_single_game();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double games_per_sec = num_games / elapsed.count();
    double avg_game_length = static_cast<double>(total_moves) / num_games;
    std::cout << "Played " << num_games << " games in " << elapsed.count() << " seconds.\n";
    std::cout << "Games per second: " << games_per_sec << std::endl;
    std::cout << "Average game length (half-moves): " << avg_game_length << std::endl;
}

} // namespace mystuff

int main() {
    mystuff::play_multiple_games(100);
    return 0;
}
