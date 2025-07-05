#include "leela_board_wrapper.h"
#include "chess/position.h"
#include <cassert>

class LeelaBoardWrapperImpl {
public:
    lczero::Position position;
    LeelaBoardWrapperImpl() : position(lczero::ChessBoard::kStartposBoard, 0, 0) {}
};

LeelaBoardWrapper::LeelaBoardWrapper() : impl_(new LeelaBoardWrapperImpl()) {}
LeelaBoardWrapper::~LeelaBoardWrapper() { delete impl_; }

void LeelaBoardWrapper::make_move(const lczero::Move& move) {
    impl_->position = lczero::Position(impl_->position, move);
}

bool LeelaBoardWrapper::is_game_over() const {
    auto moves = impl_->position.GetBoard().GenerateLegalMoves();
    // Game is over if no legal moves or 50-move rule
    if (moves.empty()) return true;
    if (impl_->position.GetRule50Ply() >= 100) return true;
    return false;
}

std::string LeelaBoardWrapper::result() const {
    auto moves = impl_->position.GetBoard().GenerateLegalMoves();
    if (!moves.empty() && impl_->position.GetRule50Ply() < 100) return "*";
    if (impl_->position.GetBoard().IsUnderCheck( )) {
        return impl_->position.IsBlackToMove() ? "1-0" : "0-1";
    }
    return "1/2-1/2";
}

std::vector<lczero::Move> LeelaBoardWrapper::legal_moves() const {
    return impl_->position.GetBoard().GenerateLegalMoves();
}
