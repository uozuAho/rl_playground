#include "bot_fight.h"
#include "torch_eval_approximator.h"

int main() {
    // mystuff::bot_fight(5);
    mystuff::EvalApproximator ea;
    ea.train_and_test_value_network(100, 10, 1);
    return 0;
}
