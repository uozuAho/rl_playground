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
public class AzSelfPlayer2 : IDisposable
{
    public readonly BlockingCollection<IChessGame> DoneQueue = new();

    private Task[] _tasks = [];
    private readonly ILogger _logger;

    private readonly BlockingCollection<SelfPlayGame> _batchQueue = new();
    private readonly BlockingCollection<(SelfPlayGame[], Tensor)> _evalQueue = new(4);
    private readonly BlockingCollection<(SelfPlayGame[], (Tensor, Tensor))> _unbatchQueue = new(8);
    private readonly BlockingCollection<SelfPlayGame> _advanceQueue = new();
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

    public AzSelfPlayer2(
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
        _tasks = [Task.Run(Advance), Task.Run(Batch), Task.Run(Eval), Task.Run(Unbatch), Task.Run(Unbatch)];
    }

    public void StopAndWait()
    {
        _logger.Debug("stop requested");
        _stopRequested = true;
        if (_gamesInProgress == 0)
        {
            _logger.Debug("0 games in progress, completing advance queue");
            _advanceQueue.CompleteAdding();
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
        var state = new SelfPlayGame(
            new MctsNode3
            {
                Parent = null,
                Prior = 1.0,
                MoveFromParent = null,
                State = game,
            },
            _numSimulations,
            cPuct: _cPuct,
            addDirichletNoise: _addDirichletNoise,
            dirichletAlpha: _dirichletAlpha,
            dirichletEpsilon: _dirichletEpsilon
        );
        // todo: make enqueue and increment atomic?
        _advanceQueue.Add(state);
        Interlocked.Increment(ref _gamesInProgress);
    }

    public void Dispose()
    {
        _batchQueue.Dispose();
        _evalQueue.Dispose();
        _unbatchQueue.Dispose();
        _advanceQueue.Dispose();
        DoneQueue.Dispose();
    }

    private void Advance()
    {
        var metrics = TaskMetrics.StartNew(nameof(Advance));

        foreach (var sim in _advanceQueue.GetConsumingEnumerable())
        {
            metrics.StartWork();
            _logger.Debug("Advance start");

            sim.Advance();
            if (sim._simCount == 1)
                metrics.IncState();

            if (sim.IsDone)
            {
                Interlocked.Decrement(ref _gamesInProgress);
                DoneQueue.Add(sim.SearchRoot.State!);
                metrics.IncGame();
                if (_gamesInProgress == 0 && _stopRequested)
                {
                    _logger.Info($"Advance: {_gamesInProgress} games in progress, completing batch queue");
                    _batchQueue.CompleteAdding();
                }
            }
            else
            {
                metrics.StopWork();
                _batchQueue.Add(sim);
            }
        }
        _logger.Debug("Advance done");
        _batchQueue.CompleteAdding();
        _metricsQueue.Add(metrics);
    }

    private void Batch()
    {
        var batchBuf = new SelfPlayGame[_maxBatchSize];
        var bufIdx = 0;
        var metrics = TaskMetrics.StartNew(nameof(Batch));

        while (true)
        {
            _batchQueue.TryTake(out var sim, TimeSpan.FromMilliseconds(1000));
            metrics.StartWork();
            _logger.Debug("Batch start");

            if (sim != null)
                batchBuf[bufIdx++] = sim;

            if (bufIdx > 0)
            {
                var batchSize = Math.Min(_maxBatchSize, _gamesInProgress);
                if (bufIdx == batchSize)
                {
                    var batch = new SelfPlayGame[batchSize];
                    var batchStates = new IChessGame[batchSize];
                    for (var i = 0; i < batchSize; i++)
                    {
                        batch[i] = batchBuf[i];
                        batchStates[i] = batchBuf[i].SearchNode.State!;
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

            if (sim == null && _batchQueue.IsAddingCompleted)
                break;
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
            metrics.IncState(sims.Length);
            _logger.Debug("Eval stop");
            metrics.StopWork();
            _unbatchQueue.Add((sims, mpv));
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
                    sim.Peval = _net.Codec.Probdist2Dict(mp, sim.SearchNode.State!);
                    sim.Veval = v;
                }
                metrics.IncState();
                metrics.StopWork();
                _advanceQueue.Add(sim);
                metrics.StartWork();
            }
            _logger.Debug("Unbatch stop");
            metrics.StopWork();
        }
        _logger.Debug("Unbatch done");
        _advanceQueue.CompleteAdding();
        _metricsQueue.Add(metrics);
    }
}

public sealed record MctsNode3
{
    public MctsNode3? Parent { get; init; }
    public double Prior { get; internal set; }
    public Move? MoveFromParent { get; init; }
    public Dictionary<Move, MctsNode3>? Children;
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

public sealed class SelfPlayGame
{
    public bool IsDone;

    public MctsNode3 SearchRoot { get; private set; }
    public MctsNode3 SearchNode { get; private set; }
    internal double? TerminalValue;
    public MoveProbs? Peval;
    public double? Veval;

    public string RootFen { get; set; }
    internal int _simCount;
    private readonly int _simLimit;

    private readonly double _cPuct;
    private readonly bool _addDirichletNoise;
    private readonly double _dirichletAlpha;
    private readonly double _dirichletEpsilon;

    public SelfPlayGame(
        MctsNode3 searchRoot,
        int numSimulations,
        double cPuct,
        bool addDirichletNoise,
        double dirichletAlpha,
        double dirichletEpsilon
    )
    {
        Debug.Assert(searchRoot.State != null);
        SearchRoot = searchRoot;
        RootFen = searchRoot.State.Fen();
        SearchRoot.State = CodingAdventureChessGame.FromFen(RootFen);
        SearchNode = SearchRoot;
        _simLimit = numSimulations;
        _cPuct = cPuct;
        _addDirichletNoise = addDirichletNoise;
        _dirichletAlpha = dirichletAlpha;
        _dirichletEpsilon = dirichletEpsilon;
    }

    private void ResetSearchNode()
    {
        SearchRoot.State = CodingAdventureChessGame.FromFen(RootFen);
        SearchNode = SearchRoot;
        TerminalValue = null;
        Peval = null;
        Veval = null;
    }

    private void ResetSearch()
    {
        Debug.Assert(SearchRoot.State != null);
        SearchRoot = new MctsNode3 { State = SearchRoot.State, Prior = 1.0 };
        RootFen = SearchRoot.State.Fen();
        ResetSearchNode();
        _simCount = 0;
    }

    /// <summary>
    /// Progress the game to completion, or to next state requiring eval
    /// </summary>
    public void Advance()
    {
        DoGreedyBestMove();
        // if (SearchRoot.Children == null)
        // {
        //     DoTreePol();
        // }
        // else
        // {
        //     ExpandAndBackprop();
        //     if (_simCount == _simLimit)
        //     {
        //         ResetSearchNode();
        //         DoBestMove();
        //         ResetSearch();
        //     }
        //     if (!IsDone)
        //     {
        //         DoTreePol();
        //     }
        // }
        //
        // _simCount++;
    }

    private void DoGreedyBestMove()
    {
        if (Peval != null)
        {
            var bestMove = Peval.MaxBy(x => x.Value).Key;
            SearchRoot.State!.MakeMove(bestMove);
            SearchNode = SearchRoot;
            IsDone = SearchRoot.State.IsGameOver();
        }
    }

    private void DoTreePol()
    {
        ResetSearchNode();

        while (SearchNode.Children?.Count > 0 && !SearchNode.IsTerminal)
        {
            SearchNode = SearchNode.Children.Values.MaxBy(c => c.Puct(_cPuct))!;
            SearchNode.State = SearchNode.Parent!.State!;
            SearchNode.State.MakeMove(SearchNode.MoveFromParent!.Value);
            SearchNode.IsTerminal = SearchNode.State.IsGameOver();
        }

        if (SearchNode.IsTerminal)
        {
            var gameState = SearchNode.State!.GameStatus();
            var turn = SearchNode.State.Turn();
            var movedLast = turn == Color.White ? Color.Black : Color.White;
            var winner = gameState.Winner;

            if (winner == null)
                TerminalValue = 0.0;
            else
            {
                TerminalValue = winner == movedLast ? 1.0 : -1.0;
            }
        }
        else
        {
            var legalMoves = SearchNode.State!.LegalMoves().ToArray();
            SearchNode.Children = new Dictionary<Move, MctsNode3>(legalMoves.Length);
            for (var i = 0; i < legalMoves.Length; i++)
            {
                var move = legalMoves[i];
                SearchNode.Children[move] = new MctsNode3
                {
                    Parent = SearchNode,
                    MoveFromParent = move,
                };
            }
        }
    }

    private void ExpandAndBackprop()
    {
        Debug.Assert(TerminalValue.HasValue || Veval.HasValue);

        if (TerminalValue == null)
        {
            Debug.Assert(Peval != null);
            Veval = -Veval;

            if (ReferenceEquals(SearchNode, SearchRoot) && _addDirichletNoise)
            {
                Maths.AddDirichletNoiseInPlace(Peval, _dirichletAlpha, _dirichletEpsilon);
            }

            if (SearchNode.Children != null)
            {
                foreach (var kv in SearchNode.Children)
                {
                    var (move, child) = kv;
                    child.Prior = Peval[move];
                }
            }
        }

        var value = TerminalValue ?? Veval!.Value;

        var node = SearchNode;
        while (node != null)
        {
            if (node.Parent == null)
            {
                node = SearchRoot;
            }
            node.Visits++;
            node.TotalValue += value;
            node = node.Parent;
            value = -value;
        }
    }

    private void DoBestMove()
    {
        Debug.Assert(_simCount == _simLimit);
        Debug.Assert(SearchRoot.State != null);
        Debug.Assert(SearchRoot.Children != null);
        var move = SearchRoot.Children.Values.MaxBy(x => x.Visits)!.MoveFromParent;
        SearchRoot.State.MakeMove(move!.Value);
        IsDone = SearchRoot.State.IsGameOver();
    }
}
