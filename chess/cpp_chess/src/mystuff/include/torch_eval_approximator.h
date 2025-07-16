#pragma once
#include <torch/torch.h>

namespace mystuff {

struct ValueNetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::Dropout dropout{nullptr};

    ValueNetImpl() {
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
TORCH_MODULE(ValueNet);

class EvalApproximator {
public:
    EvalApproximator();
    void train_and_test_value_network(
        int n_train,
        int n_test,
        int epochs,
        int batch_size=32,
        double lr=1e-3
    );
private:
    ValueNet net_;
    torch::Device device_ = torch::kCPU;
    std::tuple<std::vector<std::vector<float>>, std::vector<float>>
        generate_random_positions(int n_positions);
    float normalize_eval(int eval);
};
} // namespace mystuff
