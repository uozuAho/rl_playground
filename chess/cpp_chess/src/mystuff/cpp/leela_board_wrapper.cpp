#include "leela_board_wrapper.h"
// #include "external/lc0/src/chess/board.h" // Uncomment when lc0 is available

LeelaBoardWrapper::LeelaBoardWrapper() {
    // board_ = new lczero::Board();
}

void LeelaBoardWrapper::reset() {
    // *board_ = lczero::Board();
}

void LeelaBoardWrapper::make_move(const lczero::Move& move) {
    // board_->make_move(move);
}

const lczero::ChessBoard& LeelaBoardWrapper::board() const {
    // return *board_;
    throw std::runtime_error("Not implemented");
}

bool LeelaBoardWrapper::is_game_over() const {
    // return lczero::is_game_over(*board_);
    return false;
}

std::string LeelaBoardWrapper::result() const {
    // return lczero::result(*board_);
    return "*";
}

std::vector<lczero::Move> LeelaBoardWrapper::legal_moves() const {
    // return lczero::legal_moves(*board_);
    return {};
}
