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
/// Best I've got so far ~ 26k states/sec with
/// NN 2 48
/// one job each apart from unbatcher, 2x unbatchers
/// batch size ~40-50
/// eval util: 84% (nvtop reports ~75% GPU)
/// unbatch util: 32% (x2)
/// larger batches slow things down, and reduce eval utilisation
///
/// jobs:
/// - batch for eval: game -> (games, Tensor)
/// - eval: (games, Tensor) -> (games, (Tensor, Tensor))
/// - unbatch: (games, (Tensor, Tensor)) -> (game, policy)
/// - make moves: (game, policy) -> game
/// </summary>
public class ExperimentSaturateGpu
{
    private const int numGames = 4000;
    private const int maxBatchSize = 40;
    private static readonly BlockingCollection<IChessGame> BatchQueue = new(numGames);
    private static readonly BlockingCollection<(IChessGame[], Tensor)> EvalQueue = new(4);
    private static readonly BlockingCollection<(IChessGame[], (Tensor, Tensor))> UnbatchQueue = new(
        4
    );
    private static readonly BlockingCollection<(IChessGame, Dictionary<Move, float>)> MoveQueue =
        new(numGames);
    private static readonly BlockingCollection<TaskMetrics> MetricsQueue = new();
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
            Task.Run(Unbatch),
            Task.Run(MakeMove),
        };

        Task.WaitAll(tasks);

        for (var i = 0; i < tasks.Length; i++)
        {
            var m = MetricsQueue.Take();
            m.PrintSummary();
        }
    }

    private static void Batch()
    {
        var batchBuf = new IChessGame[maxBatchSize];
        var bufIdx = 0;
        var metrics = TaskMetrics.StartNew(nameof(Batch));

        foreach (var game in BatchQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            batchBuf[bufIdx++] = game;
            var batchSize = Math.Min(maxBatchSize, _gamesInProgress);
            if (bufIdx == batchSize)
            {
                var batch = new IChessGame[batchSize];
                batchBuf.CopyTo(batch, 0);
                var arr = Net.Codec.States2Array(batch);
                var arrT = from_array(arr).to(CUDA);
                metrics.IncState(batchSize);
                metrics.StopWork();
                EvalQueue.Add((batch, arrT));
                bufIdx = 0;
            }
        }
        EvalQueue.CompleteAdding();

        MetricsQueue.Add(metrics);
    }

    private static void Eval()
    {
        var metrics = TaskMetrics.StartNew(nameof(Eval));

        foreach (var (games, tArr) in EvalQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            var mpv = Net.Forward(tArr);
            metrics.IncState(games.Length);
            metrics.StopWork();
            UnbatchQueue.Add((games, mpv));
        }
        UnbatchQueue.CompleteAdding();

        MetricsQueue.Add(metrics);
    }

    private static void Unbatch()
    {
        var metrics = TaskMetrics.StartNew(nameof(Unbatch));

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

        MetricsQueue.Add(metrics);
    }

    private static void MakeMove()
    {
        var metrics = TaskMetrics.StartNew(nameof(MakeMove));

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
                metrics.StopWork();
            }
            else
            {
                var (move, _) = mp.MaxBy(x => x.Value);
                game.MakeMove(move);
                metrics.IncState();
                metrics.StopWork();
                BatchQueue.Add(game);
            }
        }

        MetricsQueue.Add(metrics);
    }
}
