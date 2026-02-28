using cschess.game;

namespace cschess.agents.test;

internal class UniformEvaluator : IEvaluator
{
    public IEnumerable<(Dictionary<Move, float>, float)> BatchEval(IEnumerable<IChessGame> games)
    {
        return from chessGame in games
            select chessGame.LegalMoves().ToList() into moves
            let prob = 1.0f / moves.Count
            select (moves.ToDictionary(x => x, _ => prob), 0.0f);
    }
}
