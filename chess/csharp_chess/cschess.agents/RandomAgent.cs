using cschess.csutils;
using cschess.game;

namespace cschess.agents;

public class RandomAgent : IChessAgent
{
    private readonly Random _random = new();

    public MyMove NextMove(IChessGame game, TimeSpan timeout)
    {
        return _random.Choice(game.LegalMoves());
    }
}
