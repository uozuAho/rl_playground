using cschess.game;

namespace cschess.agents;

public interface IChessAgent
{
    MyMove NextMove(IChessGame game) => NextMove(game, TimeSpan.FromMilliseconds(10));

    MyMove NextMove(IChessGame game, TimeSpan timeout);
}
