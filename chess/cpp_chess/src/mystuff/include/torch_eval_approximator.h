#pragma once
#include <torch/torch.h>

namespace mystuff {
struct ValueNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    ValueNetImpl(int input_size, int hidden_size=16) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, 1));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
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
