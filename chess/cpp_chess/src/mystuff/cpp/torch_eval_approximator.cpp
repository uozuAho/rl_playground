
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <algorithm>
#include "torch_eval_approximator.h"
#include "leela_board_wrapper.h"
#include "eval.h"

namespace mystuff {

EvalApproximator::EvalApproximator() :
    net_(ValueNet()),
    device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    net_->to(device_);
}

// Encode board as a flat vector (12x64 one-hot for pieces + 5 extras)
static std::vector<float> board2vec(const LeelaBoardWrapper& board) {
    // 12 piece types (6 per color) x 64 squares
    std::vector<float> features(12 * 64, 0.0f);
    // Piece type order: white P N B R Q K, black P N B R Q K
    for (int sq = 0; sq < 64; ++sq) {
        auto piece_opt = board.piece_at(lczero::Square::FromIdx(sq));
        if (!piece_opt.has_value()) continue;
        int color = board.color_at(lczero::Square::FromIdx(sq));
        int type = piece_opt.value().idx;
        int idx = -1;
        if (color == LeelaBoardWrapper::WHITE) idx = (type - 1);
        else idx = 6 + (type - 1);
        if (idx >= 0 && idx < 12) features[idx * 64 + sq] = 1.0f;
    }
    // Add 5 extras: turn, castling, etc. (dummy for now)
    features.resize(12 * 64 + 5, 0.0f);
    features[12 * 64] = (board.turn() == LeelaBoardWrapper::WHITE) ? 1.0f : 0.0f;
    return features;
}

// Encode board as an 8x8x8 float tensor (piece_layer, rows, cols)
static torch::Tensor board2tensor(const LeelaBoardWrapper& board) {
    torch::Tensor state = torch::zeros({8, 8, 8}, torch::kFloat32);
    for (int i = 0; i < 64; ++i) {
        int row = i / 8;
        int col = i % 8;
        auto sq = lczero::Square::FromIdx(i);
        auto piece_opt = board.piece_at(sq);
        if (!piece_opt.has_value()) continue;
        int color = board.color_at(sq);
        assert(color != 0);
        auto pieceIdx = piece_opt.value().idx;
        state[pieceIdx][row][col] = static_cast<float>(color);
    }
    float fullmove = static_cast<float>(board.fullmoveCount());
    if (fullmove > 0) {
        state[6].fill_(1.0f / fullmove);
    }
    if (board.turn() == LeelaBoardWrapper::WHITE) {
        state[6][0].fill_(1.0f);
    } else {
        state[6][0].fill_(-1.0f);
    }
    state[7].fill_(1.0f);
    return state;
}

// Normalize evaluation to [-1, 1]
float EvalApproximator::normalize_eval(int eval) {
    return std::tanh(eval / 3000.0f);
}

// Generate 2 vectors of length n: (board tensors, normalised values)
std::tuple<std::vector<torch::Tensor>, std::vector<float>>
EvalApproximator::generate_random_positions(int n_positions)
{
    std::vector<torch::Tensor> boardVecs;
    std::vector<float> normVals;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < n_positions; ++i) {
        LeelaBoardWrapper board;
        int n_moves = std::uniform_int_distribution<>(5, 50)(gen);
        for (int j = 0; j < n_moves; ++j) {
            auto moves = board.legal_moves();
            if (moves.empty() || board.is_game_over()) break;
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            board.make_move(moves[dist(gen)]);
        }
        int value = evaluate_board(board);
        boardVecs.push_back(board2tensor(board).to(device_));
        normVals.push_back(normalize_eval(value));
    }
    return std::tuple(boardVecs, normVals);
}

void EvalApproximator::train_and_test_value_network(
    int n_train,
    int n_test,
    int epochs,
    int batch_size,
    double lr
)
{
    std::cout << "Generating " << n_train << " training positions...\n";
    auto [train_boards, train_values] = generate_random_positions(n_train);

    std::cout << "Generating " << n_test << " test positions...\n";
    auto [test_boards, test_values] = generate_random_positions(n_test);

    auto to_tensor1d = [](const std::vector<float>& data) {
        return torch::from_blob((float*)data.data(), {(long)data.size(), 1}).clone();
    };
    torch::Tensor X_train = torch::stack(test_boards);
    torch::Tensor y_train = to_tensor1d(train_values).to(device_);
    torch::Tensor X_test = torch::stack(test_boards);
    torch::Tensor y_test = to_tensor1d(test_values).to(device_);

    std::cout << "X_train shape: " << X_train.sizes() << ", y_train shape: " << y_train.sizes() << std::endl;

    torch::optim::Adam optimizer(net_->parameters(), torch::optim::AdamOptions(lr));
    auto criterion = torch::nn::MSELoss();

    std::cout << "Training for " << epochs << " epochs. Batch size " << batch_size << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        net_->train();
        float epoch_loss = 0.0f;
        for (int i = 0; i < X_train.size(0); i += batch_size) {
            int end = std::min(i + batch_size, (int)X_train.size(0));
            auto batch_X = X_train.slice(0, i, end);
            auto batch_y = y_train.slice(0, i, end);
            optimizer.zero_grad();
            auto pred = net_->forward(batch_X);
            auto loss = criterion(pred, batch_y);
            loss.backward();
            optimizer.step();
            epoch_loss += loss.item<float>() * (end - i);
        }
        epoch_loss /= X_train.size(0);
        if ((epoch+1) % 2 == 0) {
            std::cout << "Epoch " << (epoch+1) << "/" << epochs << ", Loss: " << epoch_loss << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ns = end - start;
    auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(elapsed_ns);
    std::cout << "Done training in " << elapsed_s << std::endl;

    net_->eval();
    auto pred = net_->forward(X_test).detach().cpu().squeeze();
    auto y_true = y_test.cpu().squeeze();
    float mse = torch::mse_loss(pred, y_true).item<float>();
    float mae = torch::linalg_norm(pred - y_true, 1).item<float>() / y_true.size(0);
    float corr = torch::corrcoef(torch::stack({pred, y_true}))[0][1].item<float>();
    std::cout << "Test MSE: " << mse << ", MAE: " << mae << ", Corr: " << corr << std::endl;
}

} // namespace mystuff
