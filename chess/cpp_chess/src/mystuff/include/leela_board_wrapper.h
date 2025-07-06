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

    lczero::PieceType piece_at(lczero::Square square) const;
private:
    LeelaBoardWrapperImpl* impl_;
};
