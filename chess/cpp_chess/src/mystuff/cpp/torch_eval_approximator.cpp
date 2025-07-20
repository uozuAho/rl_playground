
#include <torch/torch.h>
#include <iostream>
#include <fstream>
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
    std::vector<torch::Tensor> boardTs;
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
        boardTs.push_back(board2tensor(board).to(device_));
        normVals.push_back(normalize_eval(value));
    }
    return std::tuple(boardTs, normVals);
}

std::tuple<std::vector<torch::Tensor>, std::vector<float>>
EvalApproximator::read_positions_from_csv(const std::string& filename)
{
    std::vector<torch::Tensor> boardTs;
    std::vector<float> values;
    std::ifstream infile(filename);

    if (not infile) {
        throw std::ios_base::failure("file does not exist");
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string fen, value_str;
        if (!std::getline(ss, fen, ',')) continue;
        if (!std::getline(ss, value_str)) continue;
        float value = std::stof(value_str);
        auto board = LeelaBoardWrapper::from_fen(fen);
        auto myscore = normalize_eval(evaluate_board(board));
        if (std::abs(myscore - value) > 0.01) {
            std::cout << "score mismatch: " << fen << ", cpp: " << myscore << ", py: " << value << std::endl;
        }
        boardTs.push_back(board2tensor(board).to(device_));
        values.push_back(value);
    }
    return std::tuple(boardTs, values);
}

void EvalApproximator::train_and_test_value_network(
    const std::tuple<std::vector<torch::Tensor>, std::vector<float>>& data,
    int epochs,
    int batch_size,
    double lr
)
{
    const auto& boards = std::get<0>(data);
    const auto& values = std::get<1>(data);
    size_t n = boards.size();
    if (n == 0 || values.size() != n) {
        std::cerr << "Empty or mismatched data for training." << std::endl;
        return;
    }

    // Shuffle indices for random split
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    size_t n_train = n * 8 / 10;
    size_t n_test = n - n_train;

    std::vector<torch::Tensor> train_boards, test_boards;
    std::vector<float> train_values, test_values;
    for (size_t i = 0; i < n; ++i) {
        if (i < n_train) {
            train_boards.push_back(boards[indices[i]]);
            train_values.push_back(values[indices[i]]);
        } else {
            test_boards.push_back(boards[indices[i]]);
            test_values.push_back(values[indices[i]]);
        }
    }

    auto to_tensor1d = [](const std::vector<float>& data) {
        return torch::from_blob((float*)data.data(), {(long)data.size(), 1}).clone();
    };
    torch::Tensor X_train = torch::stack(train_boards);
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

    // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    // 0: no correlation
    // +1,-1: perfect +/- correlation
    float corr = torch::corrcoef(torch::stack({pred, y_true}))[0][1].item<float>();

    std::cout << "Test MSE: " << mse << ", MAE: " << mae << ", Corr: " << corr << std::endl;
}

} // namespace mystuff
