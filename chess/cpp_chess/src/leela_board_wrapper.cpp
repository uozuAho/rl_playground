#include "leela_board_wrapper.h"
// #include "external/lc0/src/chess/board.h" // Uncomment when lc0 is available

LeelaBoardWrapper::LeelaBoardWrapper() {
    // board_ = new lcz::Board();
}

void LeelaBoardWrapper::reset() {
    // *board_ = lcz::Board();
}

void LeelaBoardWrapper::make_move(const lcz::Move& move) {
    // board_->make_move(move);
}

const lcz::Board& LeelaBoardWrapper::board() const {
    // return *board_;
    throw std::runtime_error("Not implemented");
}

bool LeelaBoardWrapper::is_game_over() const {
    // return lcz::is_game_over(*board_);
    return false;
}

std::string LeelaBoardWrapper::result() const {
    // return lcz::result(*board_);
    return "*";
}

std::vector<lcz::Move> LeelaBoardWrapper::legal_moves() const {
    // return lcz::legal_moves(*board_);
    return {};
}
