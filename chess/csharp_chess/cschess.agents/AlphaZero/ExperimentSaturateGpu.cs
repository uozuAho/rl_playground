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
/// - batch for eval: game -> (games, Tensor)
/// - eval: (games, Tensor) -> (games, (Tensor, Tensor))
/// - unbatch: (games, (Tensor, Tensor)) -> (game, policy)
/// - make moves: (game, policy) -> game
/// </summary>
public class ExperimentSaturateGpu
{
    private const int numGames = 1;
    private const int maxBatchSize = 1;
    private static readonly BlockingCollection<IChessGame> BatchQueue = new(numGames);
    private static readonly BlockingCollection<(IChessGame[], Tensor)> EvalQueue = new(numGames / maxBatchSize);
    private static readonly BlockingCollection<(IChessGame[], (Tensor, Tensor))> UnbatchQueue = new(numGames / maxBatchSize);
    private static readonly BlockingCollection<(IChessGame, Dictionary<Move, float>)> MoveQueue = new(numGames);
    private static readonly ResNet Net = new(2, 48, CUDA);
    private static int _gamesInProgress;

    public static void EvaluateSaturateGpu()
    {
        Net.Eval();

        for (var i = 0; i < numGames; i++)
        {
            BatchQueue.Add(CodingAdventureChessGame.StandardGame());
            _gamesInProgress++;
        }

        var tasks = new[]
        {
            Task.Run(Batch),
            Task.Run(Eval),
            Task.Run(Unbatch),
            Task.Run(MakeMove),
        };

        Task.WaitAll(tasks);
    }

    private static void Batch()
    {
        var batchBuf = new IChessGame[maxBatchSize];
        var bufIdx = 0;
        var metrics = TaskMetrics.StartNew();

        foreach (var game in BatchQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            batchBuf[bufIdx++] = game;
            var batchSize = Math.Min(maxBatchSize, _gamesInProgress);
            if (bufIdx == batchSize)
            {
                var arr = Net.Codec.States2Array(batchBuf);
                var arrT = from_array(arr).to(CUDA);
                metrics.IncState(batchSize);
                metrics.StopWork();
                EvalQueue.Add((batchBuf, arrT));
                bufIdx = 0;
            }
        }
        EvalQueue.CompleteAdding();

        metrics.PrintSummary(nameof(Batch));
    }

    private static void Eval()
    {
        var metrics = TaskMetrics.StartNew();

        foreach (var (games, tArr) in EvalQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            var mpv = Net.Forward(tArr);
            UnbatchQueue.Add((games, mpv));
            metrics.IncState(games.Length);
            metrics.StopWork();
        }
        UnbatchQueue.CompleteAdding();

        metrics.PrintSummary(nameof(Eval));
    }

    private static void Unbatch()
    {
        var metrics = TaskMetrics.StartNew();

        foreach (var (games, pvs) in UnbatchQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            foreach (var (game, pv) in games.Zip(Net.NnHeadsToPv(pvs.Item1, pvs.Item2)))
            {
                var (mp, _) = pv;
                var mpd = Net.Codec.Probdist2Dict(mp, game);
                metrics.IncState();
                metrics.StopWork();
                MoveQueue.Add((game, mpd));
            }
        }
        MoveQueue.CompleteAdding();

        metrics.PrintSummary(nameof(Unbatch));
    }

    private static void MakeMove()
    {
        var metrics = TaskMetrics.StartNew();

        foreach (var (game, mp) in MoveQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();

            if (game.IsGameOver())
            {
                metrics.IncGame();
                Interlocked.Decrement(ref _gamesInProgress);
                if (_gamesInProgress == 0)
                {
                    BatchQueue.CompleteAdding();
                }
            }
            else
            {
                var (move, _) = mp.MaxBy(x => x.Value);
                game.MakeMove(move);
                BatchQueue.Add(game);
                metrics.IncState();
            }

            metrics.StopWork();
        }

        metrics.PrintSummary(nameof(MakeMove));
    }
}

internal class TaskMetrics
{
    private readonly Stopwatch _sw;
    private TimeSpan _workStarted;
    private TimeSpan _workTime;
    private int _games;
    private int _states;

    private TaskMetrics(Stopwatch sw)
    {
        _sw = sw;
    }

    public static TaskMetrics StartNew()
    {
        return new TaskMetrics(Stopwatch.StartNew());
    }

    public void StartWork() => _workStarted = _sw.Elapsed;
    public void StopWork() => _workTime += _sw.Elapsed - _workStarted;
    public void IncGame() => _games += 1;
    public void IncState() => _states += 1;
    public void IncState(int nStates) => _states += nStates;

    public void PrintSummary(string name)
    {
        var totalTime = _sw.Elapsed;
        var gamesPerSec = _games / totalTime.TotalSeconds;
        var statesPerSec = _states / totalTime.TotalSeconds;
        var util = _workTime / totalTime;
        Console.WriteLine($"{name}: {_games} games, {_states} states in {totalTime}");
        Console.WriteLine($"{name}: {gamesPerSec:F2} games/sec, {statesPerSec:F2} states/sec");
        Console.WriteLine($"{name}: utilisation: {util:F2}");
    }
}
