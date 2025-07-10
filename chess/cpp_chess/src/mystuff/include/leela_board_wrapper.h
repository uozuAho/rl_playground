#pragma once
#include <vector>
#include <string>
#include <optional>

#include "chess/types.h"
#include "chess/board.h"

namespace mystuff {

class LeelaBoardWrapperImpl;
class LeelaBoardWrapper {
public:
    const static int WHITE;
    const static int BLACK;

    LeelaBoardWrapper();
    ~LeelaBoardWrapper();

    static LeelaBoardWrapper from_fen(const std::string& fen);

    void make_move(const lczero::Move& move);

    int turn() const;
    std::vector<lczero::Move> legal_moves() const;
    std::string result() const;
    bool is_game_over() const;
    std::optional<lczero::PieceType> piece_at(lczero::Square square) const;
    int color_at(lczero::Square square) const;
private:
    LeelaBoardWrapperImpl* impl_;
};

} // namespace mystuff
