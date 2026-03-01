using System.Diagnostics;
using TorchSharp.Modules;

namespace cschess.agents.AlphaZero;

using static TorchSharp.torch;

public class Az
{
    public static void Train()
    {
        const int numIterations = 1;
        const int nParallelGames = 2;
        var net = new ResNet(2, 48, CUDA);
        var optimiser = new Adam(net.ModelParams());
        var gameTimes = new List<TimeSpan>();
        var learnTimes = new List<TimeSpan>();
        var totalSamples = 0;

        var stopwatch = new Stopwatch();
        for (var i = 0; i < numIterations; i++)
        {
            List<GameSample> samples;
            TimeSpan playTime;
            using (no_grad())
            {
                stopwatch.Start();
                samples = Player.SelfPlayGames(
                    net,
                    nGames: nParallelGames,
                    nMctsSims: 60,
                    cPuct: 2.0,
                    temperature: 1.25,
                    dirichletAlpha: 0.3,
                    dirichletEpsilon: 0.25
                ).ToList();
                playTime = stopwatch.Elapsed;
                totalSamples += samples.Count;
            }
            var (ploss, vloss) = Learner.UpdateNet(net, optimiser, samples, false);
            var learnTime = stopwatch.Elapsed - playTime;
            gameTimes.Add(playTime);
            learnTimes.Add(learnTime);
            Console.WriteLine($"played {nParallelGames} games, {totalSamples} steps in {playTime}");
            Console.WriteLine($"ploss, vloss: {ploss}, {vloss}");
            var avgGameTimeS = gameTimes.Select(x => x.TotalSeconds).Average();
            var avgLearnTimeS = learnTimes.Select(x => x.TotalSeconds).Average();
            var gamesPerSec = nParallelGames / avgGameTimeS;
            var playStepsPerSec = totalSamples / avgGameTimeS;
            Console.WriteLine($"{gamesPerSec} games/sec, {playStepsPerSec} steps/sec, avg learn: {avgLearnTimeS}");
        }
    }
}
