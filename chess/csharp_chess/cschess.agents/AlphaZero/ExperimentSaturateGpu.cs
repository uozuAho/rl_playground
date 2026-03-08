using System.Collections.Concurrent;
using System.Diagnostics;
using cschess.game;
using static TorchSharp.torch;

namespace cschess.agents.AlphaZero;

public class ExperimentSaturateGpu
{
    private const int numGames = 100;
    private static BlockingCollection<IChessGame> toEvalQueue = new(numGames);
    private static BlockingCollection<(IChessGame, (float[], float))> evaldQueue = new(numGames);
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

    private static void AdvanceState()
    {
        var games = 0;
        var states = 0;
        var sw = Stopwatch.StartNew();
        var workingTime = TimeSpan.Zero;

        foreach (var (game, mpv) in evaldQueue.GetConsumingEnumerable())
        {
            var sww = Stopwatch.StartNew();

            if (game.IsGameOver())
            {
                toEvalQueue.CompleteAdding();
                games++;
            }
            else
            {
                var (mp, _) = mpv;
                var mpd = net.Codec.Probdist2Dict(mp, game);
                var (move, _) = mpd.MaxBy(x => x.Value);
                game.MakeMove(move);
                toEvalQueue.Add(game);
                states++;
            }

            workingTime += sww.Elapsed;
        }

        var totalTime = sw.Elapsed;
        var gamesPerSec = games / totalTime.TotalSeconds;
        var statesPerSec = states / totalTime.TotalSeconds;
        var util = workingTime / totalTime;
        Console.WriteLine($"Player: played {games} games, {states} states in {totalTime}");
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
            var mpv = net.Pv([game]).Single();
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
