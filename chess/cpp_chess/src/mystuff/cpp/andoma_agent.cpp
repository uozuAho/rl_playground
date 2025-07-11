#include <limits>
#include <vector>
#include <algorithm>

#include "andoma_agent.h"
#include "eval.h"

namespace mystuff {

static std::vector<lczero::Move> get_ordered_moves(const LeelaBoardWrapper& board) {
    // TODO: Implement move ordering (captures/promotions first)
    return board.legal_moves();
}

static constexpr double MATE_SCORE = 1000000000.0;
static constexpr double MATE_THRESHOLD = 999000000.0;

static double minimax(
    int depth,
    const LeelaBoardWrapper& board,
    double alpha,
    double beta,
    bool is_maximising_player)
{
    if (depth == 0 || board.is_game_over() ) {
        return evaluate_board(board);
    }

    double best_move = is_maximising_player
        ? -std::numeric_limits<double>::infinity()
        : std::numeric_limits<double>::infinity();

    auto moves = get_ordered_moves(board);

    for (const auto& move : moves) {
        auto newBoard = board.copy();
        newBoard.make_move(move);
        double value = minimax(depth - 1, newBoard, alpha, beta, !is_maximising_player);
        // TODO: Mate scoring (?)
        if (is_maximising_player) {
            best_move = std::max(best_move, value);
            alpha = std::max(alpha, best_move);
            if (beta <= alpha) {
                break;
            }
        } else {
            best_move = std::min(best_move, value);
            beta = std::min(beta, best_move);
            if (beta <= alpha) {
                break;
            }
        }
    }

    return best_move;
}

static lczero::Move minimax_root(int depth, const LeelaBoardWrapper& board) {
    bool maximize = board.turn() == LeelaBoardWrapper::WHITE;

    double best_value = maximize
        ? -std::numeric_limits<double>::infinity()
        : std::numeric_limits<double>::infinity();

    lczero::Move best_move;

    auto moves = get_ordered_moves(board);

    for (const auto& move : moves) {
        auto newBoard = board.copy();
        newBoard.make_move(move);

        double value = minimax(
            depth - 1,
            newBoard,
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            !maximize
        );

        if ((maximize && value > best_value) || (!maximize && value < best_value)) {
            best_value = value;
            best_move = move;
        }
    }

    return best_move;
}

AndomaAgent::AndomaAgent(int search_depth) : search_depth_(search_depth) {}

lczero::Move AndomaAgent::select_move(const LeelaBoardWrapper& board) {
    return minimax_root(search_depth_, board);
}

} // namespace mystuff
