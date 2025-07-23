using cschess.csutils;
using cschess.game;

namespace cschess.agents;

public class RandomAgent : IChessAgent
{
    private readonly Random _random = new();

    public Move NextMove(IChessGame game)
    {
        return _random.Choice(game.LegalMoves());
    }
}
