using cschess.csutils;
using cschess.game;
using Shouldly;

namespace cschess.agents.test;

public class PmctsTests
{
    [Fact]
    public void it_runs()
    {
        const int numGames = 2;
        const int numSims = 200;
        var games = Enumerable
            .Range(0, numGames)
            .Select(_ => CodingAdventureChessGame.StandardGame())
            .Cast<IChessGame>()
            .ToList();

        var startFens = games.Select(x => x.Fen()).ToList();

        var roots = new ParallelMcts(games, new UniformEvaluator(), numSims).Run();

        foreach (var root in roots)
        {
            root.Visits.ShouldBe(numSims);
            root.Children.ShouldNotBeNull();
            var cprobs = root.Children.Values.Select(x => x.Prior);
            Maths.IsProbDist(cprobs).ShouldBe(true);
        }

        games.Select(x => x.Fen()).ShouldBe(startFens, "should not modify games");
    }

    [Fact]
    public void chooses_checkmates()
    {
        var boardMoves = new[]
        {
            ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8"),
            ("4r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1", "e8e1"),
        };

        var games = boardMoves
            .Select(b => CodingAdventureChessGame.FromFen(b.Item1))
            .Cast<IChessGame>()
            .ToList();
        var expectedMoves = boardMoves.Select(bm => bm.Item2).ToList();

        var roots = new ParallelMcts(games, new UniformEvaluator(), 100).Run();
        var maxVisitMoves = roots.Select(x =>
            x.Children?.Values.MaxBy(c => c.Visits)!.MoveFromParent!.Value.ToUci()
        );
        maxVisitMoves.ShouldBe(expectedMoves);
    }
}
