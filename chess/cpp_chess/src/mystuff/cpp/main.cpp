#include "chess/board.h"
#include "bot_fight.h"
#include "torch_eval_approximator.h"

int main() {
    lczero::InitializeMagicBitboards();
    // mystuff::bot_fight(5);
    mystuff::EvalApproximator ea;
    // auto data = ea.read_positions_from_csv("pymieches.csv");
    auto data = ea.generate_random_positions(5000);
    ea.train_and_test_value_network(data, 50);
    return 0;
}
