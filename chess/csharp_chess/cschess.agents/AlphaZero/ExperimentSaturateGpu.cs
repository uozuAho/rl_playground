using System.Collections.Concurrent;
using System.Diagnostics;
using cschess.game;
using static TorchSharp.torch;

namespace cschess.agents.AlphaZero;

public class ExperimentSaturateGpu
{
    private const int numGames = 1;
    private static BlockingCollection<IChessGame> toEvalQueue = new(numGames + 3);
    private static BlockingCollection<(IChessGame, (Tensor, Tensor))> evaldQueue = new(numGames + 3);
    static ResNet net = new(2, 48, CUDA);

    public static void EvaluateSaturateGpu()
    {
        net.Eval();
        var player = Task.Run(AdvanceState);
        var moveEvaler = Task.Run(EvalJob);
        for (var i = 0; i < numGames; i++)
        {
            toEvalQueue.Add(CodingAdventureChessGame.StandardGame());
        }

        player.Wait();
        moveEvaler.Wait();
    }

    private static void PushToEval(IChessGame game)
    {

    }

    private static void AdvanceState()
    {
        var gameCount = 0;
        var stateCount = 0;
        var sw = Stopwatch.StartNew();
        var workingTime = TimeSpan.Zero;

        foreach (var (game, mpv) in evaldQueue.GetConsumingEnumerable())
        {
            var sww = Stopwatch.StartNew();

            if (game.IsGameOver())
            {
                toEvalQueue.CompleteAdding();
                gameCount++;
            }
            else
            {
                var (mp, _) = net.NnHeadsToPv(mpv.Item1, mpv.Item2).Single();
                var mpd = net.Codec.Probdist2Dict(mp, game);
                var (move, _) = mpd.MaxBy(x => x.Value);
                game.MakeMove(move);
                toEvalQueue.Add(game);
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

        foreach (var game in toEvalQueue.GetConsumingEnumerable())
        {
            // todo: convert to/from array etc on producer side?
            // todo: convert to/from tensor on producer side?
            // todo: transfer to cuda on producer side?

            var sww = Stopwatch.StartNew();
            var arr = net.Codec.States2Array([game]);
            var tArr = from_array(arr).to(CUDA);
            var mpv = net.Forward(tArr);
            evaldQueue.Add((game, mpv));
            states++;
            workingTime += sww.Elapsed;
        }
        evaldQueue.CompleteAdding();

        var totalTime = sw.Elapsed;
        var statesPerSec = states / totalTime.TotalSeconds;
        var util = workingTime / totalTime;
        Console.WriteLine($"Evaler: evaled {states} states in {totalTime}");
        Console.WriteLine($"Evaler: {statesPerSec:F2} states/sec");
        Console.WriteLine($"Evaler: utilisation: {util:F2}");
    }
}
