#pragma once
#include <vector>
#include <string>

#include "chess/types.h"
#include "chess/board.h"

class LeelaBoardWrapperImpl;
class LeelaBoardWrapper {
public:
    LeelaBoardWrapper();
    ~LeelaBoardWrapper();
    void make_move(const lczero::Move& move);
    bool is_game_over() const;
    std::string result() const;
    std::vector<lczero::Move> legal_moves() const;
    // Returns true if it is black's turn to move, false if white's turn
    bool is_black_to_move() const;
    // Returns 0 for white, 1 for black
    int turn() const;
    // Returns the color of the player to move as a string ("white" or "black")
    std::string color() const;
private:
    LeelaBoardWrapperImpl* impl_;
};
