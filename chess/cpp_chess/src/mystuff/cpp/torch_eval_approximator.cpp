#include <torch/torch.h>
#include "torch_eval_approximator.h"

namespace mystuff {

void EvalApproximator::doSimpleTorchAction() {
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;
}

} // namespace mystuff
