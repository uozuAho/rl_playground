// Shameless copy of https://github.com/healeycodes/andoma/blob/main/evaluate.py
// ported to C++ by copilot.

// this module implement's Tomasz Michniewski's Simplified Evaluation Function
// https://www.chessprogramming.org/Simplified_Evaluation_Function
// note that the board layouts have been flipped and the top left square is A1

#include "leela_board_wrapper.h"
#include <vector>
#include <optional>
#include <cmath>

namespace mystuff {

constexpr uint8_t   kKnightIdx = lczero::kKnight.idx,
                    kQueenIdx = lczero::kQueen.idx,
                    kRookIdx = lczero::kRook.idx,
                    kBishopIdx = lczero::kBishop.idx,
                    kPawnIdx = lczero::kPawn.idx,
                    kKingIdx = lczero::kKing.idx;

// Piece values
// [None, Pawn, Knight, Bishop, Rook, Queen, King]
constexpr int PIECE_VALUE[] = {0, 100, 320, 330, 500, 900, 20000};

constexpr int pawnEvalWhite[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, -20, -20, 10, 10,  5,
    5, -5, -10,  0,  0, -10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0
};
constexpr int pawnEvalBlack[64] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5, -10,  0,  0, -10, -5,  5,
    5, 10, 10, -20, -20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
};
constexpr int knightEval[64] = {
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
};
constexpr int bishopEvalWhite[64] = {
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
};
constexpr int bishopEvalBlack[64] = {
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
};
constexpr int rookEvalWhite[64] = {
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
};
constexpr int rookEvalBlack[64] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0
};
constexpr int queenEval[64] = {
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
};
constexpr int kingEvalWhite[64] = {
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
};
constexpr int kingEvalBlack[64] = {
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20
};
constexpr int kingEvalEndGameWhite[64] = {
    50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30,  0,  0,  0,  0, -30, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -20, -10,  0,  0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50
};
constexpr int kingEvalEndGameBlack[64] = {
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,  0,  0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30,  0,  0,  0,  0, -30, -30,
    50, -30, -30, -30, -30, -30, -30, -50
};

int piece_square_value(
    lczero::PieceType piece_type,
    int color,
    lczero::Square square,
    bool isEndgame)
{
    const auto squareIdx = square.as_idx();

    switch (piece_type.idx) {
        case kPawnIdx:
            return color == LeelaBoardWrapper::WHITE ? pawnEvalWhite[squareIdx] : pawnEvalBlack[squareIdx];
        case kKnightIdx:
            return knightEval[squareIdx];
        case kBishopIdx:
            return color == LeelaBoardWrapper::WHITE ? bishopEvalWhite[squareIdx] : bishopEvalBlack[squareIdx];
        case kRookIdx:
            return color == LeelaBoardWrapper::WHITE ? rookEvalWhite[squareIdx] : rookEvalBlack[squareIdx];
        case kQueenIdx:
            return queenEval[squareIdx];
        case kKingIdx:
            if (isEndgame)
                return color == LeelaBoardWrapper::WHITE ? kingEvalEndGameWhite[squareIdx] : kingEvalEndGameBlack[squareIdx];
            else
                return color == LeelaBoardWrapper::WHITE ? kingEvalWhite[squareIdx] : kingEvalBlack[squareIdx];
        default:
            return 0;
    }
}

int evaluate_piece(
    lczero::PieceType piece_type,
    int color,
    lczero::Square square,
    bool isEndgame)
{
    return PIECE_VALUE[piece_type.idx] + piece_square_value(piece_type, color, square, isEndgame);
}

bool f_isEndgame(const LeelaBoardWrapper& board) {
    // TODO: Implement real logic. For now, always return false.
    return false;
}

int evaluate_board(const LeelaBoardWrapper& board) {
    int total = 0;
    bool isEndgame = f_isEndgame(board);
    // DEBUG CODE: remove me once this is working
    int piec[64] = {0};
    int colr[64] = {0};
    int vals[64] = {0};
    for (int sq = 0; sq < 64; ++sq) {
        auto square = lczero::Square::FromIdx(sq);
        auto piece_opt = board.piece_at(lczero::Square::FromIdx(sq));
        if (!piece_opt.has_value()) continue;
        auto piece_type = piece_opt.value();
        int color = board.color_at(square);
        // todo: colors are reversed (and probably pieces. note that this eval code treats 0 = A1 = top left)
        int value = evaluate_piece(piece_type, color, square, isEndgame);

        // DEBUG CODE: remove me once this is working
        if (color == LeelaBoardWrapper::BLACK)
        {
            value = -value;
        }
        piec[sq] = piece_type.idx;
        colr[sq] = color;
        vals[sq] = value;
        total += value;
    }
    return total;
}

// Evaluate a move (stub: only returns 0 for now)
float move_value(const LeelaBoardWrapper& board, int move, bool endgame) {
    // TODO: Implement move evaluation logic
    return 0.0f;
}

} // namespace mystuff
