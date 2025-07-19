/**
 * Simple greedy nn agent that makes greedy moves while training and evaluating
 */

#include <algorithm>
#include <random>
#include <iostream>

#include "leela_board_wrapper.h"
#include "greedy_chess_agent.h"

namespace mystuff {

const float REWARD_FOR_PIECE_CAPTURE = 0.001;


ExperienceReplay::ExperienceReplay(size_t capacity) : capacity_(capacity) {}

void ExperienceReplay::push(
    const torch::Tensor& state,
    const torch::Tensor& next_state,
    float reward)
{
    if (buffer_.size() >= capacity_) buffer_.pop_front();
    buffer_.push_back({state, next_state, reward});
}

std::vector<Experience> ExperienceReplay::sample(size_t batch_size) {
    size_t n = std::min(batch_size, buffer_.size());
    std::vector<size_t> indices(buffer_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    std::vector<Experience> batch;
    for (size_t i = 0; i < n; ++i) batch.push_back(buffer_[indices[i]]);
    return batch;
}

size_t ExperienceReplay::size() const { return buffer_.size(); }

GreedyChessAgent::GreedyChessAgent(
    float lr,
    float gamma,
    float tau,
    size_t batch_size
) : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      value_net_(ValueNet1()),
      target_net_(ValueNet1()),
      optimizer_(value_net_->parameters(), lr),
      gamma_(gamma),
      tau_(tau),
      batch_size_(batch_size),
      replay_buffer_(10000),
      max_grad_norm_(1.0),
      episode_count_(0)
{
    // todo: probably need this:
    // target_net_->load_state_dict(value_net_->state_dict());
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

lczero::Move GreedyChessAgent::select_move(const LeelaBoardWrapper& env) {
    assert(env.turn() == LeelaBoardWrapper::WHITE);
    auto legal_moves = env.legal_moves();
    if (legal_moves.empty()) throw std::runtime_error("No legal moves available");
    std::vector<torch::Tensor> resulting_states;
    for (const auto& move : legal_moves) {
        // perf todo: undo may be better? don't need a copy of the board
        auto result_state = env.copy();
        result_state.make_move(move);
        resulting_states.push_back(board2tensor(result_state).to(device_));
    }
    auto state_tensors = torch::stack(resulting_states, 0);
    torch::NoGradGuard no_grad;
    value_net_->eval();
    auto values = value_net_->forward(state_tensors).squeeze();
    value_net_->train();
    auto best_idx = values.argmax().item<int>();
    return legal_moves[best_idx];
}

void GreedyChessAgent::add_experience(
    const torch::Tensor& state,
    const torch::Tensor& next_state,
    float reward)
{
    replay_buffer_.push(state, next_state, reward);
}

void GreedyChessAgent::update_target_network() {
    auto params = value_net_->named_parameters();
    auto target_params = target_net_->named_parameters();
    for (auto& kv : params) {
        auto& name = kv.key();
        auto& param = kv.value();
        auto& target_param = target_params[name];
        target_param.data().copy_(tau_ * param.data() + (1.0f - tau_) * target_param.data());
    }
}

float GreedyChessAgent::train_step() {
    if (replay_buffer_.size() < batch_size_) return -1.0f;
    auto batch = replay_buffer_.sample(batch_size_);
    std::vector<torch::Tensor> states, next_states;
    std::vector<float> rewards;
    for (const auto& exp : batch) {
        states.push_back(exp.state);
        next_states.push_back(exp.next_state);
        rewards.push_back(exp.reward);
    }
    auto states_t = torch::stack(states, 0).to(device_);
    auto next_states_t = torch::stack(next_states, 0).to(device_);
    auto rewards_t = torch::tensor(rewards, torch::dtype(torch::kFloat32)).to(device_);
    auto current_values = value_net_->forward(states_t).squeeze();

    torch::Tensor target_values;
    {
        torch::NoGradGuard no_grad;
        auto next_values = target_net_->forward(next_states_t).squeeze();
        target_values = rewards_t + gamma_ * next_values;
    }
    auto loss = torch::mse_loss(current_values, target_values);
    optimizer_.zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(value_net_->parameters(), max_grad_norm_);
    optimizer_.step();
    update_target_network();
    return loss.item<float>();
}

static float reward(
    const LeelaBoardWrapper& from_state,
    const LeelaBoardWrapper& to_state,
    const int player,
    const lczero::Move move)
{
    auto result = to_state.result();

    if (result == "1-0") {
        return 1.0;
    }
    else if (result == "0-1") {
        return -1.0;
    }
    else if (result == "1/2-1/2") {
        return 0.0;
    }

    auto target_piece = from_state.piece_at(move.to());
    if (target_piece.has_value()) {
        return player * REWARD_FOR_PIECE_CAPTURE;
    }

    return 0.0;
}

void GreedyChessAgent::train_against(
    Agent& opponent,
    int n_episodes,
    float capture_reward_factor,
    int halfmove_limit,
    bool print_every
) {
    for (int episode = 0; episode < n_episodes; ++episode) {
        LeelaBoardWrapper game;
        bool done = false;
        Agent* agents[2] = {this, &opponent};
        auto prev_state = game.copy();
        auto prev_state_t = board2tensor(game).to(device_);
        std::vector<float> episode_losses;
        float episode_reward = 0.0f;
        int agentTurn = 0;

        while (!game.is_game_over()) {
            if (agentTurn == 0) assert(game.turn() == LeelaBoardWrapper::WHITE);
            else if (agentTurn == 1) assert(game.turn() == LeelaBoardWrapper::BLACK);

            auto move = agents[agentTurn]->select_move(game);
            game.make_move(move);
            auto state = board2tensor(game).to(device_);
            if (game.turn() == LeelaBoardWrapper::BLACK) {
                auto rewardd = reward(prev_state, game, LeelaBoardWrapper::WHITE, move);
                add_experience(prev_state_t, state, rewardd);
                episode_reward += rewardd;
            }
            if (game.turn() == LeelaBoardWrapper::WHITE) {
                prev_state = game.copy();
                prev_state_t = state;
            }
            float loss = train_step();
            if (loss >= 0.0f) episode_losses.push_back(loss);
            agentTurn = 1 - agentTurn;
        }

        episode_count_++;
        auto result = game.result();
        episode_wins_.push_back(result == "1-0" ? 1 : 0);
        episode_game_lengths_.push_back(game.fullmoveCount()); // todo: make this halfmove
        float avg_loss = episode_losses.empty()
            ? 0.0f
            : std::accumulate(
                episode_losses.begin(),
                episode_losses.end(),
                0.0f
              ) / episode_losses.size();
        episode_losses_.push_back(avg_loss);
        episode_rewards_.push_back(episode_reward);
        if (print_every && ((episode + 1) % 10 == 0)) {
            int recent = std::min(100, (int)episode_wins_.size());
            float win_rate = std::accumulate(
                episode_wins_.end() - recent,
                episode_wins_.end(),
                0.0f
            ) / recent;
            std::cout << "Episode " << (episode+1) << "/" << n_episodes;
            std::cout << ", Win Rate: " << win_rate << ", Avg Loss: " << avg_loss;
            std::cout << ", Game Len: " << game.fullmoveCount() << ", Reward: ";
            std::cout << episode_reward << std::endl;
        }
    }
}

} // namespace mystuff
