using System.Collections.Concurrent;
using System.Diagnostics;
using cschess.game;
using static TorchSharp.torch;

namespace cschess.agents.AlphaZero;

/// <summary>
/// Maximise games/states per second, doing greedy moves based on NN policy output.
///
/// Batching is essential for maximising GPU utilisation.
///
/// idea: jobs:
/// - make moves: (game, policy) -> game
/// - batch for eval: game -> (games, Tensor)
/// - eval: (games, Tensor) -> (games, (Tensor, Tensor))
/// - unbatch: (games, (Tensor, Tensor)) -> (game, policy)
/// </summary>
public class ExperimentSaturateGpu
{
    private const int numGames = 100;
    private static readonly BlockingCollection<(IChessGame, Tensor)> EvalQueue = new(numGames);
    private static readonly BlockingCollection<(IChessGame, (Tensor, Tensor))> PlayQueue = new(numGames);
    static ResNet net = new(2, 48, CUDA);
    private static int gamesInProgress;

    public static void EvaluateSaturateGpu()
    {
        net.Eval();
        var player = Task.Run(AdvanceState);
        var moveEvaler = Task.Run(EvalJob);
        for (var i = 0; i < numGames; i++)
        {
            PushToEval(CodingAdventureChessGame.StandardGame());
            gamesInProgress++;
        }

        player.Wait();
        moveEvaler.Wait();
    }

    private static void PushToEval(IChessGame game)
    {
        var arr = net.Codec.States2Array([game]);
        var tArr = from_array(arr).to(CUDA);
        EvalQueue.Add((game, tArr));
    }

    private static void AdvanceState()
    {
        var gameCount = 0;
        var stateCount = 0;
        var sw = Stopwatch.StartNew();
        var workingTime = TimeSpan.Zero;

        foreach (var (game, mpv) in PlayQueue.GetConsumingEnumerable())
        {
            var sww = Stopwatch.StartNew();

            if (game.IsGameOver())
            {
                gameCount++;
                gamesInProgress--;
                if (gamesInProgress == 0)
                {
                    EvalQueue.CompleteAdding();
                }
            }
            else
            {
                var (mp, _) = net.NnHeadsToPv(mpv.Item1, mpv.Item2).Single();
                var mpd = net.Codec.Probdist2Dict(mp, game);
                var (move, _) = mpd.MaxBy(x => x.Value);
                game.MakeMove(move);
                PushToEval(game);
                stateCount++;
            }

            workingTime += sww.Elapsed;
        }

        var totalTime = sw.Elapsed;
        var gamesPerSec = gameCount / totalTime.TotalSeconds;
        var statesPerSec = stateCount / totalTime.TotalSeconds;
        var util = workingTime / totalTime;
        Console.WriteLine($"Player: played {gameCount} games, {stateCount} states in {totalTime}");
        Console.WriteLine($"Player: {gamesPerSec:F2} games/sec, {statesPerSec:F2} states/sec");
        Console.WriteLine($"Player: utilisation: {util:F2}");
    }

    private static void EvalJob()
    {
        var states = 0;
        var sw = Stopwatch.StartNew();
        var workingTime = TimeSpan.Zero;

        foreach (var (game, tArr) in EvalQueue.GetConsumingEnumerable())
        {
            var sww = Stopwatch.StartNew();
            var mpv = net.Forward(tArr);
            PlayQueue.Add((game, mpv));
            states++;
            workingTime += sww.Elapsed;
        }
        PlayQueue.CompleteAdding();

        var totalTime = sw.Elapsed;
        var statesPerSec = states / totalTime.TotalSeconds;
        var util = workingTime / totalTime;
        Console.WriteLine($"Evaler: evaled {states} states in {totalTime}");
        Console.WriteLine($"Evaler: {statesPerSec:F2} states/sec");
        Console.WriteLine($"Evaler: utilisation: {util:F2}");
    }
}
