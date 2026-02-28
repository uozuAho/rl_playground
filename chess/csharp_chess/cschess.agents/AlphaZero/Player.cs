using System.Diagnostics;
using cschess.csutils;
using cschess.game;

namespace cschess.agents.AlphaZero;

public class Player
{
    public static IEnumerable<GameSample> SelfPlayGames(
        IEvaluator evaluator,
        int nGames,
        int nMctsSims,
        double cPuct,
        double temperature,
        double dirichletAlpha,
        double dirichletEpsilon
    )
    {
        var rng = new Random();
        var games = Enumerable.Range(0, nGames).Select(_ => NewChessGame()).ToList();
        var gameOvers = games.Select(_ => false).ToList();
        var trajectories = games.Select(_ => new List<GameSample>()).ToList();
        var winners = games.Select(Color? (_) => null).ToList();
        while (!gameOvers.All(x => x))
        {
            var activeIdxs = gameOvers
                .Select((gameover, idx) => (gameover, idx))
                .Where(vi => !vi.gameover)
                .Select(vi => vi.idx)
                .ToList();
            var activeGames = activeIdxs.Select(i => games[i]).ToList();
            var roots = new ParallelMcts(
                activeGames,
                evaluator,
                nMctsSims,
                cPuct,
                addDirichletNoise: true,
                dirichletAlpha,
                dirichletEpsilon
            ).Run();
            foreach (var (idx, root) in activeIdxs.Zip(roots))
            {
                var state = root.State();
                var probs = MctsProbs(root);
                Debug.Assert(probs.Count > 0);
                trajectories[idx].Add(new GameSample(state.Copy(), probs, -999));
                probs = Maths.Heat(probs, temperature);
                var move = rng.Choice(probs.Keys, probs.Values);
                games[idx].MakeMove(move);
                if (games[idx].IsGameOver())
                {
                    gameOvers[idx] = true;
                    winners[idx] = games[idx].GameStatus().Winner;
                }
            }
        }

        for (var i = 0; i < trajectories.Count; i++)
        {
            var traj = trajectories[i];
            var winner = winners[i];
            foreach (var (game, probs, _) in traj)
            {
                var reward =
                    winner == null ? 0f
                    : game.Turn() == winner ? 1.0f
                    : -1.0f;
                yield return new GameSample(game, probs, reward);
            }
        }
    }

    private static IChessGame NewChessGame() => CodingAdventureChessGame.StandardGame();

    private static Dictionary<Move, float> MctsProbs(MctsNode root)
    {
        var totalVisits = root.Children.Values.Select(c => c.Visits).Sum();
        Debug.Assert(totalVisits > 0);
        var probs = root.Children.ToDictionary(
            kv => kv.Key,
            kv => kv.Value.Visits / (float)totalVisits
        );
        Debug.Assert(Maths.IsProbDist(probs.Values));
        return probs;
    }
}

public record GameSample(IChessGame Game, Dictionary<Move, float> MoveProbs, float FinalReward);
