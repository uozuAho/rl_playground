using cschess.csutils;
using cschess.game;

namespace cschess.agents;

public class MctsAgent(
    IEvaluator evaluator,
    int nSims,
    double cPuct = 1.0,
    bool addDirichletNoise = false,
    double epsilon = 0.25,
    double alpha = 0.3
) : IChessAgent
{
    public static MctsAgent Unif(int nSims)
    {
        return new MctsAgent(new UniformBatchEval(), nSims);
    }

    public static MctsAgent RandomRollout(int nSims)
    {
        return new MctsAgent(new RandomRolloutEval(), nSims);
    }

    public Move NextMove(IChessGame game, TimeSpan timeout)
    {
        return NextMoves([game], timeout).First();
    }

    private IEnumerable<Move> NextMoves(IEnumerable<IChessGame> games, TimeSpan timeout)
    {
        var roots = new ParallelMcts(
            games.ToList(),
            evaluator,
            nSims,
            cPuct,
            addDirichletNoise,
            alpha,
            epsilon
        ).Run();
        foreach (var root in roots)
        {
            var maxVis = root.Children?.Values.MaxBy(c => c.Visits);
            if (maxVis?.MoveFromParent == null)
            {
                throw new InvalidOperationException("doh");
            }

            yield return maxVis.MoveFromParent.Value;
        }
    }
}

internal class UniformBatchEval : IEvaluator
{
    public IEnumerable<(Dictionary<Move, float>, float)> BatchEval(IEnumerable<IChessGame> games)
    {
        return from game in games
            select game.LegalMoves().ToList() into moves
            let prob = 1.0f / moves.Count
            select moves.ToDictionary(x => x, _ => prob) into probs
            select (probs, 0.0f);
    }
}

internal class RandomRolloutEval : IEvaluator
{
    public IEnumerable<(Dictionary<Move, float>, float)> BatchEval(IEnumerable<IChessGame> games)
    {
        return games.Select(EvalSingle);
    }

    private static (Dictionary<Move, float>, float) EvalSingle(IChessGame game)
    {
        var rng = new Random();
        var player = game.Turn();
        var legalMoves = game.LegalMoves().ToList();
        var moveProbs = legalMoves.ToDictionary(x => x, _ => 1.0f / legalMoves.Count);

        var gCopy = game.Copy();
        while (!gCopy.IsGameOver())
        {
            var move = rng.Choice(gCopy.LegalMoves());
            gCopy.MakeMove(move);
        }

        var gs = gCopy.GameStatus();
        var val = 0.0f;
        if (gs.Winner != null)
        {
            val = player == gs.Winner ? 1.0f : -1.0f;
        }

        return (moveProbs, val);
    }
}
