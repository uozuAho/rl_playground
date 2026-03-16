using System.Collections.Concurrent;
using System.Diagnostics;
using cschess.agents.AlphaZero;
using cschess.csutils;
using cschess.game;
using static TorchSharp.torch;

namespace cschess.agents;

using MoveProbs = Dictionary<Move, float>;

/// <summary>
/// Self plays games to completion, using MCTS with NN evaluator.
/// </summary>
public class AzSelfPlayer(
    IAzNet net,
    int numSimulations,
    int maxBatchSize,
    Device device,
    double cPuct = 1.0,
    bool addDirichletNoise = false,
    double dirichletAlpha = 0.3,
    double dirichletEpsilon = 0.25)
    : IDisposable
{
    public readonly BlockingCollection<IChessGame> DoneQueue = new();

    private Task[] _tasks = [];
    private readonly ILogger _logger = new ConsoleLogger();
    private readonly Device _device = device;

    private readonly BlockingCollection<MctsSimState2> _startSimQueue = new();
    private readonly BlockingCollection<MctsSimState2> _batchQueue = new();
    private readonly BlockingCollection<(MctsSimState2[], Tensor)> _evalQueue = new();
    private readonly BlockingCollection<(MctsSimState2[], (Tensor, Tensor))> _unbatchQueue = new();
    private readonly BlockingCollection<MctsSimState2> _finishSimQueue = new();
    private readonly BlockingCollection<MctsSimState2> _moveQueue = new();

    private int _gamesInProgress;
    private bool _stopRequested;

    public void Start()
    {
        _tasks =
        [
            Task.Run(StartSim),
            Task.Run(Batch),
            Task.Run(Eval),
            Task.Run(Unbatch),
            Task.Run(FinishSim),
            Task.Run(Move)
        ];
    }

    public void StopAndWait()
    {
        _logger.Log("stop requested");
        _stopRequested = true;
        if (_gamesInProgress == 0)
        {
            _logger.Log("0 games in progress, completing start queue");
            _startSimQueue.CompleteAdding();
        }
        Task.WaitAll(_tasks);
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
            }, numSimulations
        );
        _startSimQueue.Add(state);
    }

    private void StartSim()
    {
        foreach (var sim in _startSimQueue.GetConsumingEnumerable())
        {
            _logger.Log("StartSim");
            sim.Reset();

            while (sim.Node.Children?.Count > 0 && !sim.Node.IsTerminal)
            {
                sim.Node = sim.Node.Children.Values.MaxBy(c => c.Puct(cPuct))!;
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

                _finishSimQueue.Add(sim);
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

                _batchQueue.Add(sim);
            }
        }
        _logger.Log("StartSim done");

        _batchQueue.CompleteAdding();

        Thread.Sleep(TimeSpan.FromSeconds(1));
        _finishSimQueue.CompleteAdding();
    }

    private void Batch()
    {
        var batchBuf = new MctsSimState2[maxBatchSize];
        var bufIdx = 0;
        var metrics = TaskMetrics.StartNew(nameof(Batch));

        foreach (var sim in _batchQueue.GetConsumingEnumerable())
        {
            _logger.Log("Batch");
            metrics.StartWork();
            batchBuf[bufIdx++] = sim;
            var batchSize = Math.Min(maxBatchSize, _gamesInProgress);
            if (bufIdx == batchSize)
            {
                var batch = new MctsSimState2[batchSize];
                var batchStates = new IChessGame[batchSize];
                for (var i = 0; i < batchSize; i++)
                {
                    batch[i] = batchBuf[i];
                    batchStates[i] = batchBuf[i].Node.State!;
                }
                var batchArray = net.Codec.States2Array(batchStates);
                var batchTensor = from_array(batchArray).to(_device);
                metrics.IncState(batchSize);
                metrics.StopWork();
                _evalQueue.Add((batch, batchTensor));
                bufIdx = 0;
            }
        }
        _logger.Log("Batch done");
        _evalQueue.CompleteAdding();
    }

    private void Eval()
    {
        foreach (var (sims, simsTensor) in _evalQueue.GetConsumingEnumerable())
        {
            _logger.Log("Eval");
            var mpv = net.Forward(simsTensor);
            _unbatchQueue.Add((sims, mpv));
        }
        _logger.Log("Eval done");
        _unbatchQueue.CompleteAdding();
    }

    private void Unbatch()
    {
        foreach (var (sims, pvs) in _unbatchQueue.GetConsumingEnumerable())
        {
            _logger.Log("Unbatch");
            foreach (var (sim, pv) in sims.Zip(net.Codec.NnHeadsToPv(pvs.Item1, pvs.Item2)))
            {
                var (mp, v) = pv;
                sim.Peval = net.Codec.Probdist2Dict(mp, sim.Node.State!);
                sim.Veval = v;
                _finishSimQueue.Add(sim);
            }
        }
        _logger.Log("Unbatch done");
        _finishSimQueue.CompleteAdding();
    }

    private void FinishSim()
    {
        foreach (var sim in _finishSimQueue.GetConsumingEnumerable())
        {
            _logger.Log("FinishSim");
            Debug.Assert(sim.TerminalValue.HasValue || sim.Veval.HasValue);

            if (sim.TerminalValue == null)
            {
                Debug.Assert(sim.Peval != null);
                sim.Veval = -sim.Veval;

                if (ReferenceEquals(sim.Node, sim.Root) && addDirichletNoise)
                {
                    Maths.AddDirichletNoiseInPlace(sim.Peval, dirichletAlpha, dirichletEpsilon);
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
                _moveQueue.Add(sim);
            }
            else
            {
                _startSimQueue.Add(sim);
            }
        }
        _logger.Log("FinishSim done");
        _moveQueue.CompleteAdding();
        _startSimQueue.CompleteAdding();
    }

    private void Move()
    {
        foreach (var sim in _moveQueue.GetConsumingEnumerable())
        {
            _logger.Log("Move");
            var move = sim.Root.Children!.Values.MaxBy(x => x.Visits)!.MoveFromParent;
            sim.Root.State!.MakeMove(move!.Value);
            if (sim.Root.State.IsGameOver())
            {
                DoneQueue.Add(sim.Root.State);
                Interlocked.Decrement(ref _gamesInProgress);
                if (_gamesInProgress == 0 && _stopRequested)
                {
                    _logger.Log("Move: 0 games in progress, completing start queue");
                    _startSimQueue.CompleteAdding();
                }
            }
            else
            {
                Continue(sim.Root.State);
            }
        }
        _logger.Log("Move done");
        DoneQueue.CompleteAdding();
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

internal class TaskMetrics
{
    private readonly string _name;
    private readonly Stopwatch _sw;
    private TimeSpan _workStarted;
    private TimeSpan _workTime;
    private int _games;
    private int _states;

    private TaskMetrics(string name, Stopwatch sw)
    {
        _name = name;
        _sw = sw;
    }

    public static TaskMetrics StartNew(string name)
    {
        return new TaskMetrics(name, Stopwatch.StartNew());
    }

    public void StartWork() => _workStarted = _sw.Elapsed;
    public void StopWork() => _workTime += _sw.Elapsed - _workStarted;
    public void IncGame() => _games += 1;
    public void IncState() => _states += 1;
    public void IncState(int nStates) => _states += nStates;

    public void PrintSummary()
    {
        var totalTime = _sw.Elapsed;
        var gamesPerSec = _games / totalTime.TotalSeconds;
        var statesPerSec = _states / totalTime.TotalSeconds;
        var util = _workTime / totalTime;
        Console.WriteLine($"{_name}: {_games} games, {_states} states in {totalTime}");
        Console.WriteLine($"{_name}: {gamesPerSec:F2} games/sec, {statesPerSec:F2} states/sec");
        Console.WriteLine($"{_name}: utilisation: {util:F2}");
    }
}

interface ILogger
{
    void Log(string msg);
}

class ConsoleLogger : ILogger
{
    Stopwatch _sw = Stopwatch.StartNew();

    public void Log(string msg)
    {
        Console.WriteLine($"{_sw.ElapsedMilliseconds}: {msg}");
    }
}
