using System.Collections.Concurrent;
using System.Diagnostics;
using cschess.game;
using static TorchSharp.torch;

namespace cschess.agents.AlphaZero;

/// <summary>
/// Generate empty chess-shaped tensors and eval them as fast
/// as u can. Where 'state' = 1 chess-sized tensor,:
/// - batch size 1: 1500 states/sec
/// - batch size 20: 20k states/sec
///
/// Could possibly go even faster with some optimisation.
/// </summary>
public class ExperimentSaturateGpu2
{
    private const int numStates = 40000;
    private const int batchSize = 20;
    private static readonly BlockingCollection<Tensor> EvalQueue = new(100);
    static ResNet net = new(2, 48, CUDA);

    public static void EvaluateSaturateGpu()
    {
        net.Eval();
        var player = Task.Run(GenStates);
        var moveEvaler = Task.Run(EvalJob);

        player.Wait();
        moveEvaler.Wait();
    }

    private static void GenStates()
    {
        var stateCount = 0;
        var sw = Stopwatch.StartNew();

        for (int i = 0; i < numStates / batchSize; i++)
        {
            var fakeStates = new float[batchSize, 8, 8, 8];
            var tArr = from_array(fakeStates).to(CUDA);
            EvalQueue.Add(tArr);
            stateCount += batchSize;
        }
        EvalQueue.CompleteAdding();

        var totalTime = sw.Elapsed;
        var statesPerSec = stateCount / totalTime.TotalSeconds;
        Console.WriteLine($"Gen: {stateCount} states in {totalTime}");
        Console.WriteLine($"Gen: {statesPerSec:F2} states/sec");
    }

    private static void EvalJob()
    {
        var states = 0;
        var sw = Stopwatch.StartNew();
        var workingTime = TimeSpan.Zero;

        foreach (var tArr in EvalQueue.GetConsumingEnumerable())
        {
            var sww = Stopwatch.StartNew();
            net.Forward(tArr);
            states += batchSize;
            workingTime += sww.Elapsed;
        }

        var totalTime = sw.Elapsed;
        var statesPerSec = states / totalTime.TotalSeconds;
        var util = workingTime / totalTime;
        Console.WriteLine($"Evaler: evaled {states} states in {totalTime}");
        Console.WriteLine($"Evaler: {statesPerSec:F2} states/sec");
        Console.WriteLine($"Evaler: utilisation: {util:F2}");
    }
}
