#include "chess/board.h"
#include "bot_fight.h"
#include "torch_eval_approximator.h"

int main() {
    lczero::InitializeMagicBitboards();
    // mystuff::bot_fight(5);
    mystuff::EvalApproximator ea;
    ea.train_and_test_value_network(5000, 1000, 50);
    return 0;
}
