using System.Diagnostics;
using Chess.Core;
using cschess.game;
using Move = cschess.game.Move;

namespace cschess.agents;

public sealed class CodingAdventureAgent : IChessAgent
{
    public Move NextMove(IChessGame game, TimeSpan fromMilliseconds)
    {
        var timeout = TimeSpan.FromMilliseconds(1);
        // todo: maybe later: use opening book before search

        var cGame = game as CodingAdventureChessGame;
        Debug.Assert(cGame != null, nameof(cGame) + " != null");
        var search = new Searcher(cGame.InternalBoard);
        var move = search.StartSearch(timeout);
        return CodingAdventureChessGame.ToMyMove(move);
    }
}
