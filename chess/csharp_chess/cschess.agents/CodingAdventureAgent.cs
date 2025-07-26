using System.Diagnostics;
using Chess.Core;
using cschess.game;
using Move = cschess.game.Move;

namespace cschess.agents;

public sealed class CodingAdventureAgent : IChessAgent
{
    // public Move NextMove(IChessGame game)
    // {
    //     var search = new Searcher(game.InternalBoard);
    //     Chess.Core.Move? move = null;
    //     search.OnSearchComplete += m => move = m;
    //     Task.Run(() => search.StartSearch());
    //     Task.Delay(100).Wait();
    //     search.EndSearch();
    //     Task.Run(() =>
    //     {
    //         while (move == null)
    //         {
    //             Thread.Sleep(1);
    //         }
    //     }).Wait();
    //     Debug.Assert(move is not null);

    //     return Move.From(move.Value);

    //     // todo: maybe later: use opening book
    // }

    public Move NextMove(IChessGame game)
    {
        var search = new Searcher(game.InternalBoard);
        Chess.Core.Move? move = null;
        search.OnSearchComplete += m => move = m;

        using var cts = new CancellationTokenSource(100); // Cancel after 100ms

        var searchTask = Task.Run(() => search.StartSearch(cts.Token), cts.Token);

        try
        {
            searchTask.Wait(); // Wait for search to complete or be cancelled
        }
        catch (AggregateException ex) when (ex.InnerExceptions.Any(e => e is TaskCanceledException))
        {
            // Expected if cancelled
        }

        search.EndSearch();

        Debug.Assert(move is not null);
        return Move.From(move.Value);
    }
}
