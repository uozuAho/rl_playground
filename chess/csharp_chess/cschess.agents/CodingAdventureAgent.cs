using Chess.Core;
using cschess.game;
using Move = cschess.game.Move;

namespace cschess.agents;

public sealed class CodingAdventureAgent : IChessAgent
{
    public Move NextMove(IChessGame game)
    {
        TimeSpan timeout = TimeSpan.FromMilliseconds(1);
        // todo: maybe later: use opening book before search

        var search = new Searcher(game.InternalBoard);
        var move = search.StartSearch(timeout);
        return Move.From(move);
    }
}
