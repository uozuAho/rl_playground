#pragma once
#include <vector>
#include <string>

// Forward declaration for LeelaChessZero board
namespace lcz {
    class Board;
    struct Move;
    std::vector<Move> legal_moves(const Board& board);
    bool is_game_over(const Board& board);
    std::string result(const Board& board);
}

class LeelaBoardWrapper {
public:
    LeelaBoardWrapper();
    void reset();
    void make_move(const lcz::Move& move);
    const lcz::Board& board() const;
    bool is_game_over() const;
    std::string result() const;
    std::vector<lcz::Move> legal_moves() const;
private:
    lcz::Board* board_;
};
