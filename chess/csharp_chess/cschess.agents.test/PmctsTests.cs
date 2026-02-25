using cschess.game;
using Shouldly;

namespace cschess.agents.test;

public class PmctsTests
{
    [Fact]
    public void it_runs()
    {
        var numGames = 2;
        var numSims = 3;
        var games = Enumerable
            .Range(0, numGames)
            .Select(_ => CodingAdventureChessGame.StandardGame())
            .Cast<IChessGame>()
            .ToList();

        var roots = new ParallelMcts(games, new DummyEval(), numSims).Run();

        foreach (var root in roots)
        {
            root.Visits.ShouldBe(numSims);
            // todo: assert prob dist
        }
    }
}

internal class DummyEval : IEvaluator
{
    public IEnumerable<(Dictionary<Move, double>, double)> BatchEval(IEnumerable<IChessGame> games)
    {
        return from chessGame in games
            select chessGame.LegalMoves().ToList() into moves
            let prob = 1.0 / moves.Count
            select (moves.ToDictionary(x => x, x => prob), 0.0);
    }
}
