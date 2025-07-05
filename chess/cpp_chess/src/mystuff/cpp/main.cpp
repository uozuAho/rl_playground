#include <iostream>
#include "game.h"
#include "agent_random.h"
#include "chess/board.h"

int main() {
    lczero::InitializeMagicBitboards();

    RandomAgent white;
    RandomAgent black;
    Game game(&white, &black);
    game.play();
    std::cout << "Result: " << game.result() << std::endl;
    return 0;
}
