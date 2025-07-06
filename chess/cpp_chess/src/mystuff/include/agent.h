#pragma once
#include <vector>
#include <string>

#include "chess/types.h"
#include "chess/board.h"
#include "leela_board_wrapper.h"

namespace mystuff {

class Agent {
public:
    virtual ~Agent() = default;
    virtual lczero::Move select_move(const LeelaBoardWrapper& board) = 0;
    virtual std::string name() const = 0;
};

} // namespace mystuff
