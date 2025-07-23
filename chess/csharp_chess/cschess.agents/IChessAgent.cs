using cschess.game;

namespace cschess.agents;

public interface IChessAgent
{
    Move NextMove(IChessGame game);
}
