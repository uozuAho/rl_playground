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

std::optional<lczero::PieceType> LeelaBoardWrapper::piece_at(lczero::Square square) const
{
    assert(square >= 0 && square < 64);

    const auto board = impl_->position.GetBoard();

    if (board.pawns().get(square)) {
        return lczero::kPawn;
    } else if (board.knights().get(square)) {
        return lczero::kKnight;
    } else if (board.bishops().get(square)) {
        return lczero::kBishop;
    } else if (board.rooks().get(square)) {
        return lczero::kRook;
    } else if (board.queens().get(square)) {
        return lczero::kQueen;
    } else if (board.kings().get(square)) {
        return lczero::kKing;
    }

    return std::nullopt;
}

int LeelaBoardWrapper::color_at(lczero::Square square) const
{
    const auto board = impl_->position.GetBoard();

    if (board.ours().get(square)) {
        return LeelaBoardWrapper::WHITE;
    }

    return LeelaBoardWrapper::BLACK;
}
