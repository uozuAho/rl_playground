#include "leela_board_wrapper.h"
#include "chess/position.h"
#include <cassert>

namespace mystuff {

class LeelaBoardWrapperImpl {
public:
    lczero::Position position;
    LeelaBoardWrapperImpl() : position(lczero::ChessBoard::kStartposBoard, 0, 0) {}
};

LeelaBoardWrapper::LeelaBoardWrapper() : impl_(new LeelaBoardWrapperImpl()) {}
LeelaBoardWrapper::~LeelaBoardWrapper() { delete impl_; }

const int LeelaBoardWrapper::WHITE = 1;
const int LeelaBoardWrapper::BLACK = -1;

void LeelaBoardWrapper::make_move(const lczero::Move& move) {
    impl_->position = lczero::Position(impl_->position, move);
}

void LeelaBoardWrapper::make_move(std::string_view from, std::string_view to) {
    const auto fromSq = lczero::Square::Parse(from);
    const auto toSq = lczero::Square::Parse(to);
    const auto move = lczero::Move::White(fromSq, toSq);
    make_move(move);
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
    assert(square.as_idx() >= 0 && square.as_idx() < 64);

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

std::optional<lczero::PieceType> LeelaBoardWrapper::piece_at(std::string_view square) const
{
    return piece_at(lczero::Square::Parse(square));
}

int LeelaBoardWrapper::color_at(lczero::Square square) const
{
    const auto board = impl_->position.GetBoard();

    if (board.ours().get(square)) {
        return LeelaBoardWrapper::WHITE;
    }

    return LeelaBoardWrapper::BLACK;
}

int LeelaBoardWrapper::color_at(std::string_view square) const
{
    return color_at(lczero::Square::Parse(square));
}

LeelaBoardWrapper LeelaBoardWrapper::from_fen(const std::string& fen) {
    LeelaBoardWrapper board;
    board.impl_->position = lczero::Position::FromFen(fen);
    return board;
}

int LeelaBoardWrapper::turn() const
{
    return impl_->position.IsBlackToMove() ? BLACK : WHITE;
}

LeelaBoardWrapper LeelaBoardWrapper::copy() const {
    // perf: ChessBoard(board) exists, may be faster if we don't need position
    LeelaBoardWrapper new_wrapper;
    new_wrapper.impl_->position = impl_->position;
    return new_wrapper;
}

std::string LeelaBoardWrapper::fen() const {
    return lczero::BoardToFen(impl_->position.GetBoard());
}

} // namespace mystuff
