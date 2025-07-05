#pragma once
#include <vector>
#include <string>

// Forward declaration for LeelaChessZero board
namespace lczero {
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
    void make_move(const lczero::Move& move);
    const lczero::Board& board() const;
    bool is_game_over() const;
    std::string result() const;
    std::vector<lczero::Move> legal_moves() const;
private:
    lczero::Board* board_;
};
