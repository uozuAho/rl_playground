using cschess.game;

namespace cschess.agents;

public interface IChessAgent
{
    Move NextMove(IChessGame game) => NextMove(game, TimeSpan.FromMilliseconds(10));

    Move NextMove(IChessGame game, TimeSpan timeout);
}
