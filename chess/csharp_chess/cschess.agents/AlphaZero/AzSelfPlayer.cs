using System.Collections.Concurrent;
using System.Diagnostics;
using cschess.csutils;
using cschess.game;
using MoreLinq;
using static TorchSharp.torch;

namespace cschess.agents.AlphaZero;

using MoveProbs = Dictionary<Move, float>;

/// <summary>
/// Self plays games to completion, using MCTS with NN evaluator.
/// </summary>
public class AzSelfPlayer : IDisposable
{
    public readonly BlockingCollection<IChessGame> DoneQueue = new();

    private Task[] _tasks = [];
    private readonly ILogger _logger;

    private readonly BlockingCollection<MctsSimState2> _startSimQueue = new();
    private readonly BlockingCollection<MctsSimState2> _batchQueue = new();
    private readonly BlockingCollection<(MctsSimState2[], Tensor)> _evalQueue = new(4);
    private readonly BlockingCollection<(MctsSimState2[], (Tensor, Tensor))> _unbatchQueue = new(8);
    private readonly BlockingCollection<MctsSimState2> _finishSimQueue = new();
    private readonly BlockingCollection<MctsSimState2> _moveQueue = new();
    private readonly BlockingCollection<TaskMetrics> _metricsQueue = new();

    private int _gamesInProgress;
    private bool _stopRequested;
    private readonly IAzNet _net;
    private readonly int _numSimulations;
    private readonly int _maxBatchSize;
    private readonly int _unbatchSize;
    private readonly Device _device;
    private readonly double _cPuct;
    private readonly bool _addDirichletNoise;
    private readonly double _dirichletAlpha;
    private readonly double _dirichletEpsilon;

    public AzSelfPlayer(
        IAzNet net,
        int numSimulations,
        int maxBatchSize,
        int unbatchSize,
        Device device,
        double cPuct = 1.0,
        bool addDirichletNoise = false,
        double dirichletAlpha = 0.3,
        double dirichletEpsilon = 0.25,
        LogLevel logLevel = LogLevel.Info
    )
    {
        if (maxBatchSize % unbatchSize != 0)
            throw new ArgumentException("maxBatchSize must be a multiple of unbatchSize");

        _net = net;
        _numSimulations = numSimulations;
        _maxBatchSize = maxBatchSize;
        _unbatchSize = unbatchSize;
        _device = device;
        _cPuct = cPuct;
        _addDirichletNoise = addDirichletNoise;
        _dirichletAlpha = dirichletAlpha;
        _dirichletEpsilon = dirichletEpsilon;
        _logger = logLevel == LogLevel.None ? new NullLogger() : new ConsoleLogger(logLevel);
    }

    public void Start()
    {
        _tasks =
        [
            Task.Run(StartSim),
            Task.Run(Batch),
            Task.Run(Eval),
            Task.Run(Unbatch),
            Task.Run(Unbatch),
            Task.Run(FinishSim),
            Task.Run(Move),
        ];
    }

    public void StopAndWait()
    {
        _logger.Debug("stop requested");
        _stopRequested = true;
        if (_gamesInProgress == 0)
        {
            _logger.Debug("0 games in progress, completing start queue");
            _startSimQueue.CompleteAdding();
        }
        Task.WaitAll(_tasks);

        for (var i = 0; i < _tasks.Length; i++)
        {
            var metrics = _metricsQueue.Take();
            _logger.Info(metrics.Summary());
        }
    }

    public void Enqueue(IChessGame game)
    {
        Continue(game);
        Interlocked.Increment(ref _gamesInProgress);
    }

    public void Dispose()
    {
        _startSimQueue.Dispose();
        _batchQueue.Dispose();
        _evalQueue.Dispose();
        _unbatchQueue.Dispose();
        _finishSimQueue.Dispose();
        _moveQueue.Dispose();
        DoneQueue.Dispose();
    }

    private void Continue(IChessGame game)
    {
        var state = new MctsSimState2(
            new MctsNode2
            {
                Parent = null,
                Prior = 1.0,
                MoveFromParent = null,
                State = game,
            },
            _numSimulations
        );
        _startSimQueue.Add(state);
    }

    private void StartSim()
    {
        var metrics = TaskMetrics.StartNew(nameof(StartSim));

        foreach (var sim in _startSimQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            metrics.IncState();
            _logger.Debug("StartSim start");
            sim.Reset();

            while (sim.Node.Children?.Count > 0 && !sim.Node.IsTerminal)
            {
                sim.Node = sim.Node.Children.Values.MaxBy(c => c.Puct(_cPuct))!;
                sim.Node.State = sim.Node.Parent!.State!;
                sim.Node.State.MakeMove(sim.Node.MoveFromParent!.Value);
                sim.Node.IsTerminal = sim.Node.State.IsGameOver();
            }

            if (sim.Node.IsTerminal)
            {
                var gameState = sim.Node.State!.GameStatus();
                var turn = sim.Node.State.Turn();
                var movedLast = turn == Color.White ? Color.Black : Color.White;
                var winner = gameState.Winner;

                if (winner == null)
                    sim.TerminalValue = 0.0;
                else
                {
                    sim.TerminalValue = winner == movedLast ? 1.0 : -1.0;
                }
            }
            else
            {
                var legalMoves = sim.Node.State!.LegalMoves().ToArray();
                sim.Node.Children = new Dictionary<Move, MctsNode2>(legalMoves.Length);
                for (var i = 0; i < legalMoves.Length; i++)
                {
                    var move = legalMoves[i];
                    sim.Node.Children[move] = new MctsNode2
                    {
                        Parent = sim.Node,
                        MoveFromParent = move,
                    };
                }
            }

            // note: we add terminal states to the batcher, even though they don't
            // need to be evaluated. The alternative is to send terminal states to
            // the finish sim queue, however then you need to somehow tell the batcher
            // that fewer states need evaluating
            _logger.Debug("StartSim stop");
            metrics.StopWork();
            _batchQueue.Add(sim);
        }
        _logger.Debug("StartSim done");

        _batchQueue.CompleteAdding();
        _metricsQueue.Add(metrics);

        Thread.Sleep(TimeSpan.FromSeconds(1));
        _finishSimQueue.CompleteAdding();
    }

    private void Batch()
    {
        var batchBuf = new MctsSimState2[_maxBatchSize];
        var bufIdx = 0;
        var metrics = TaskMetrics.StartNew(nameof(Batch));

        foreach (var sim in _batchQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            _logger.Debug("Batch start");
            batchBuf[bufIdx++] = sim;
            var batchSize = Math.Min(_maxBatchSize, _gamesInProgress);
            if (bufIdx == batchSize)
            {
                var batch = new MctsSimState2[batchSize];
                var batchStates = new IChessGame[batchSize];
                for (var i = 0; i < batchSize; i++)
                {
                    batch[i] = batchBuf[i];
                    batchStates[i] = batchBuf[i].Node.State!;
                }
                var batchArray = _net.Codec.States2Array(batchStates);
                var batchTensor = from_array(batchArray).to(_device);
                metrics.IncState(batchSize);
                _logger.Debug("Batch stop");
                metrics.StopWork();
                _evalQueue.Add((batch, batchTensor));
                bufIdx = 0;
            }
        }
        _logger.Debug("Batch done");
        _evalQueue.CompleteAdding();
        _metricsQueue.Add(metrics);
    }

    private void Eval()
    {
        var metrics = TaskMetrics.StartNew(nameof(Eval));

        foreach (var (sims, simsTensor) in _evalQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            _logger.Debug("Eval start");
            var mpv = _net.Forward(simsTensor);
            var (p, v) = mpv;

            var batchNum = 0;
            foreach (var simBatch in sims.Batch(_unbatchSize))
            {
                var mpvSlice = (
                    p.narrow(0, batchNum * _unbatchSize, simBatch.Length),
                    v.narrow(0, batchNum * _unbatchSize, simBatch.Length)
                );
                batchNum++;
                metrics.StopWork();
                _unbatchQueue.Add((simBatch, mpvSlice));
                metrics.StartWork();
            }

            metrics.IncState(sims.Length);
            _logger.Debug("Eval stop");
            metrics.StopWork();
        }
        _logger.Debug("Eval done");
        _unbatchQueue.CompleteAdding();
        _metricsQueue.Add(metrics);
    }

    private void Unbatch()
    {
        var metrics = TaskMetrics.StartNew(nameof(Unbatch));

        foreach (var (sims, pvs) in _unbatchQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            _logger.Debug("Unbatch start");
            foreach (var (sim, pv) in sims.Zip(_net.Codec.NnHeadsToPv(pvs.Item1, pvs.Item2)))
            {
                if (!sim.TerminalValue.HasValue)
                {
                    var (mp, v) = pv;
                    sim.Peval = _net.Codec.Probdist2Dict(mp, sim.Node.State!);
                    sim.Veval = v;
                }
                metrics.IncState();
                metrics.StopWork();
                _finishSimQueue.Add(sim);
                metrics.StartWork();
            }
            _logger.Debug("Unbatch stop");
            metrics.StopWork();
        }
        _logger.Debug("Unbatch done");
        _finishSimQueue.CompleteAdding();
        _metricsQueue.Add(metrics);
    }

    private void FinishSim()
    {
        var metrics = TaskMetrics.StartNew(nameof(FinishSim));

        foreach (var sim in _finishSimQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            _logger.Debug("FinishSim start");
            metrics.IncState();
            Debug.Assert(sim.TerminalValue.HasValue || sim.Veval.HasValue);

            if (sim.TerminalValue == null)
            {
                Debug.Assert(sim.Peval != null);
                sim.Veval = -sim.Veval;

                if (ReferenceEquals(sim.Node, sim.Root) && _addDirichletNoise)
                {
                    Maths.AddDirichletNoiseInPlace(sim.Peval, _dirichletAlpha, _dirichletEpsilon);
                }

                if (sim.Node.Children != null)
                {
                    foreach (var kv in sim.Node.Children)
                    {
                        var (move, child) = kv;
                        child.Prior = sim.Peval[move];
                    }
                }
            }

            var value = sim.TerminalValue ?? sim.Veval!.Value;

            var node = sim.Node;
            while (node != null)
            {
                if (node.Parent == null)
                {
                    node = sim.Root;
                }
                node.Visits++;
                node.TotalValue += value;
                node = node.Parent;
                value = -value;
            }

            if (++sim.SimCount == sim.SimLimit)
            {
                sim.Reset();
                metrics.StopWork();
                _logger.Debug("FinishSim stop");
                _moveQueue.Add(sim);
            }
            else
            {
                metrics.StopWork();
                _logger.Debug("FinishSim stop");
                _startSimQueue.Add(sim);
            }
        }
        _logger.Debug("FinishSim done");
        _moveQueue.CompleteAdding();
        _startSimQueue.CompleteAdding();
        _metricsQueue.Add(metrics);
    }

    private void Move()
    {
        var metrics = TaskMetrics.StartNew(nameof(Move));

        foreach (var sim in _moveQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            _logger.Debug("Move start");
            var move = sim.Root.Children!.Values.MaxBy(x => x.Visits)!.MoveFromParent;
            sim.Root.State!.MakeMove(move!.Value);
            metrics.IncState();
            if (sim.Root.State.IsGameOver())
            {
                metrics.IncGame();
                metrics.StopWork();
                DoneQueue.Add(sim.Root.State);
                Interlocked.Decrement(ref _gamesInProgress);
                if (_gamesInProgress == 0 && _stopRequested)
                {
                    _logger.Debug("Move: 0 games in progress, completing start queue");
                    _startSimQueue.CompleteAdding();
                }
            }
            else
            {
                _logger.Debug("Move stop");
                metrics.StopWork();
                Continue(sim.Root.State);
            }
        }
        _logger.Debug("Move done");
        DoneQueue.CompleteAdding();
        _metricsQueue.Add(metrics);
    }
}

internal record MctsNode2
{
    public MctsNode2? Parent { get; init; }
    public double Prior { get; internal set; }
    public Move? MoveFromParent { get; init; }
    public Dictionary<Move, MctsNode2>? Children;
    public int Visits;
    public double TotalValue;

    public IChessGame? State;

    public bool IsTerminal;

    public override string ToString()
    {
        return $"vists: {Visits}, puct1: {Puct(1.0):F2}, p: {Prior:F2}, tv: {TotalValue:F2}";
    }

    private double Value() => Visits == 0 ? 0 : TotalValue / Visits;

    internal double Puct(double cPuct)
    {
        var v = 0.0;
        if (Parent != null)
        {
            v = Math.Sqrt(Parent.Visits) / (1 + Visits);
        }

        return Value() + cPuct * Prior * v;
    }
}

internal class MctsSimState2
{
    internal MctsNode2 Root { get; set; }
    internal MctsNode2 Node { get; set; }
    internal int SimCount { get; set; }
    internal int SimLimit { get; init; }
    internal double? TerminalValue;
    internal MoveProbs? Peval;
    internal double? Veval;
    private string RootFen { get; }

    public MctsSimState2(MctsNode2 root, int numSimulations)
    {
        Debug.Assert(root.State != null);
        Root = root;
        RootFen = root.State.Fen();
        Node = ResetNode();
        SimLimit = numSimulations;
    }

    internal void Reset()
    {
        Node = ResetNode();
        TerminalValue = null;
        Peval = null;
        Veval = null;
    }

    private MctsNode2 ResetNode()
    {
        var newNode = Root;
        newNode.State = CodingAdventureChessGame.FromFen(RootFen);
        return newNode;
    }
}

public enum LogLevel
{
    Debug = 0,
    Info,
    None,
}

internal interface ILogger
{
    void Debug(string msg);
    void Info(string msg);
}

internal class ConsoleLogger(LogLevel level) : ILogger
{
    Stopwatch _sw = Stopwatch.StartNew();

    public void Debug(string msg)
    {
        if (level <= LogLevel.Debug)
            Console.WriteLine($"{_sw.ElapsedMilliseconds}: {msg}");
    }

    public void Info(string msg)
    {
        if (level <= LogLevel.Info)
            Console.WriteLine($"{_sw.ElapsedMilliseconds}: {msg}");
    }
}

internal class NullLogger : ILogger
{
    public void Debug(string msg) { }

    public void Info(string msg) { }
}
