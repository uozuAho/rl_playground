using System.Diagnostics;
using Chess.Core;
using cschess.game;
using Move = cschess.game.Move;

namespace cschess.agents;

public sealed class CodingAdventureAgent : IChessAgent
{
    public Move NextMove(IChessGame game)
    {
        var search = new Searcher(game.InternalBoard);
        Chess.Core.Move? move = null;
        search.OnSearchComplete += m => move = m;
        Task.Run(() => search.StartSearch());
        Task.Delay(100).Wait();
        search.EndSearch();
        Task.Run(() =>
        {
            while (move == null)
            {
                Thread.Sleep(1);
            }
        }).Wait();
        Debug.Assert(move is not null);

        return Move.From(move.Value);

        // todo: maybe later: use opening book
    }
}
