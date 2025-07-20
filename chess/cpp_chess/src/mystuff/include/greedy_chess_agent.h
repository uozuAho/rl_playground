#pragma once
#include <torch/torch.h>
#include <vector>
#include <deque>

#include "chess/types.h"

#include "leela_board_wrapper.h"
#include "agent.h"

namespace mystuff {

struct ValueNet1Impl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::Dropout dropout{nullptr};

    ValueNet1Impl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 32, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 3).padding(1)));

        fc1 = register_module("fc1", torch::nn::Linear(32 * 8 * 8, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 1));

        dropout = register_module("dropout", torch::nn::Dropout(0.3));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));

        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = dropout->forward(x);
        x = torch::relu(fc2->forward(x));
        x = torch::tanh(fc3->forward(x));
        return x;
    }
};
TORCH_MODULE(ValueNet1);


struct Experience {
    torch::Tensor state;
    torch::Tensor next_state;
    float reward;
};

class ExperienceReplay {
public:
    ExperienceReplay(size_t capacity = 10000);
    void push(const torch::Tensor& state, const torch::Tensor& next_state, float reward);
    std::vector<Experience> sample(size_t batch_size);
    size_t size() const;
private:
    std::deque<Experience> buffer_;
    size_t capacity_;
};

class GreedyChessAgent : public Agent {
public:
    GreedyChessAgent(
        float lr = 1e-3,
        float gamma = 0.99,
        float tau = 0.001,
        size_t batch_size = 32
    );

    void train_against(
        Agent& opponent,
        int n_episodes,
        float capture_reward_factor = 0.0,
        int halfmove_limit = -1,
        bool print_every = true
    );

    lczero::Move select_move(const LeelaBoardWrapper& board) override;

    void add_experience(
        const torch::Tensor& state,
        const torch::Tensor& next_state,
        float reward);

    float train_step();

    std::string name() const override { return "GreedyChessAgent"; }
private:
    torch::Device device_;
    ValueNet1 value_net_;
    ValueNet1 target_net_;
    torch::optim::Adam optimizer_;
    float gamma_;
    float tau_;
    size_t batch_size_;
    ExperienceReplay replay_buffer_;
    float max_grad_norm_;
    // Training metrics
    std::vector<int> episode_wins_;
    std::vector<int> episode_game_lengths_;
    std::vector<float> episode_losses_;
    std::vector<float> episode_rewards_;
    int episode_count_;
    void update_target_network();
};

} // namespace mystuff
