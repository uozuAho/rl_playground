
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <algorithm>
#include "torch_eval_approximator.h"
#include "leela_board_wrapper.h"
#include "eval.h"

namespace mystuff {

EvalApproximator::EvalApproximator() :
    net_(ValueNet(input_size_)),
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

// Generate n random positions by random playouts
std::vector<std::vector<float>> EvalApproximator::generate_random_positions(int n_positions) {
    std::vector<std::vector<float>> positions;
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
        positions.push_back(board2vec(board));
    }
    return positions;
}

// Normalize evaluation to [-1, 1]
float EvalApproximator::normalize_eval(int eval) {
    return std::tanh(eval / 3000.0f);
}

void EvalApproximator::train_and_test_value_network(
    int n_train,
    int n_test,
    int epochs,
    int batch_size,
    double lr
)
{
    // todo: convert these to a dataset generation function, then split
    // into test/train
    std::cout << "Generating training positions...\n";
    auto train_positions = generate_random_positions(n_train);
    std::vector<float> train_targets;
    for (const auto& feat : train_positions) {
        LeelaBoardWrapper board = LeelaBoardWrapper();
        int eval = evaluate_board(board);
        train_targets.push_back(normalize_eval(eval));
    }

    std::cout << "Generating test positions...\n";
    auto test_positions = generate_random_positions(n_test);
    std::vector<float> test_targets;
    for (const auto& feat : test_positions) {
        LeelaBoardWrapper board = LeelaBoardWrapper();
        int eval = evaluate_board(board);
        test_targets.push_back(normalize_eval(eval));
    }

    // Convert to tensors
    auto to_tensor = [](const std::vector<std::vector<float>>& data) {
        return torch::from_blob((float*)data.data(), {(long)data.size(), (long)data[0].size()}).clone();
    };
    auto to_tensor1d = [](const std::vector<float>& data) {
        return torch::from_blob((float*)data.data(), {(long)data.size(), 1}).clone();
    };
    torch::Tensor X_train = to_tensor(train_positions).to(device_);
    torch::Tensor y_train = to_tensor1d(train_targets).to(device_);
    torch::Tensor X_test = to_tensor(test_positions).to(device_);
    torch::Tensor y_test = to_tensor1d(test_targets).to(device_);

    torch::optim::Adam optimizer(net_->parameters(), torch::optim::AdamOptions(lr));
    auto criterion = torch::nn::MSELoss();

    std::cout << "Training for " << epochs << " epochs...\n";
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

    // Evaluate
    net_->eval();
    auto pred = net_->forward(X_test).detach().cpu().squeeze();
    auto y_true = y_test.cpu().squeeze();
    float mse = torch::mse_loss(pred, y_true).item<float>();
    float mae = torch::linalg_norm(pred - y_true, 1).item<float>() / y_true.size(0);
    float corr = torch::corrcoef(torch::stack({pred, y_true}))[0][1].item<float>();
    std::cout << "Test MSE: " << mse << ", MAE: " << mae << ", Corr: " << corr << std::endl;
}

} // namespace mystuff
