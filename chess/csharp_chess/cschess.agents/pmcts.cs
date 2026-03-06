using System.Diagnostics;
using cschess.csutils;
using cschess.game;

namespace cschess.agents;

using MoveProbs = Dictionary<Move, float>;

public interface IEvaluator
{
    IEnumerable<(MoveProbs, float)> BatchEval(IEnumerable<IChessGame> games);
}

public record MctsNode
{
    public MctsNode? Parent { get; init; }
    public double Prior { get; internal set; }
    public Move? MoveFromParent { get; init; }
    public Dictionary<Move, MctsNode>? Children;
    public int Visits;
    public double TotalValue;

    // todo: make this private
    internal IChessGame? _state;

    internal bool _isTerminal;

    public override string ToString()
    {
        return $"vists: {Visits}, puct1: {Puct(1.0):F2}, p: {Prior:F2}, tv: {TotalValue:F2}";
    }

    public IChessGame State()
    {
        if (_state != null)
            return _state;

        Debug.Assert(Parent != null);
        _state = Parent.State();
        Debug.Assert(MoveFromParent.HasValue);
        _state.MakeMove(MoveFromParent.Value);

        return _state;
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

internal class MctsSimState
{
    internal MctsNode Root { get; set; }
    internal MctsNode Node { get; set; }
    internal double? TerminalValue;
    internal MoveProbs? Peval;
    internal double? Veval;
    private IChessGame RootState { get; }

    public MctsSimState(MctsNode root)
    {
        Root = root;
        RootState = Root.State().Copy();
        Node = ResetNode();
    }

    internal void Reset()
    {
        Node = ResetNode();
        TerminalValue = null;
        Peval = null;
        Veval = null;
    }

    private MctsNode ResetNode()
    {
        var newNode = Root;
        newNode._state = RootState.Copy();
        return newNode;
    }
}

public class ParallelMcts(
    List<IChessGame> states,
    IEvaluator evaluator,
    int numSimulations,
    double cPuct = 1.0,
    bool addDirichletNoise = false,
    double dirichletAlpha = 0.3,
    double dirichletEpsilon = 0.25
)
{
    private int _simCount;
    private List<MctsSimState> _sims = [];

    public List<MctsNode> Run()
    {
        _sims = states
            .Select(state => new MctsSimState(
                new MctsNode
                {
                    Parent = null,
                    Prior = 1.0,
                    MoveFromParent = null,
                    _state = state,
                }
            ))
            .ToList();

        while (_simCount < numSimulations)
        {
            StartSim();
            Eval();
            FinishSim();
            _simCount++;
        }

        return _sims.Select(s => s.Root).ToList();
    }

    private void StartSim()
    {
        foreach (var sim in _sims)
        {
            sim.Reset();

            while (sim.Node.Children?.Count > 0 && !sim.Node._isTerminal)
            {
                sim.Node = sim.Node.Children.Values.MaxBy(c => c.Puct(cPuct))!;
                // node's state is the same ref as its parent. ew this code is yuck
                // I'm calculating state here at the latest possible moment for perf
                // reasons. Only the tree policy node needs to have its state calculated,
                // thus we can reuse the parent state instead of copying it
                sim.Node._state = sim.Node.Parent!.State();
                sim.Node._state.MakeMove(sim.Node.MoveFromParent!.Value);
                sim.Node._isTerminal = sim.Node._state.IsGameOver();
            }

            if (sim.Node._isTerminal)
            {
                var gameState = sim.Node.State().GameStatus();
                var turn = sim.Node.State().Turn();
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
                var legalMoves = sim.Node.State().LegalMoves().ToArray();
                sim.Node.Children = new Dictionary<Move, MctsNode>(legalMoves.Length);
                for (var i = 0; i < legalMoves.Length; i++)
                {
                    var move = legalMoves[i];
                    sim.Node.Children[move] = new MctsNode
                    {
                        Parent = sim.Node,
                        MoveFromParent = move,
                    };
                }
            }
        }
    }

    private void Eval()
    {
        var games = _sims.Where(s => !s.Node._isTerminal).Select(s => s.Node.State()).ToList();
        if (games.Count == 0)
        {
            return;
        }
        var pvs = evaluator.BatchEval(games).ToList();
        foreach (var spv in _sims.Where(s => !s.Node._isTerminal).Zip(pvs))
        {
            var (sim, pv) = spv;
            var (p, v) = pv;
            sim.Peval = p;
            sim.Veval = v;
        }
    }

    private void FinishSim()
    {
        foreach (var sim in _sims)
        {
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

                // var legalMoves = sim.Node.State().LegalMoves().ToArray();
                // sim.Node.Children = new Dictionary<Move, MctsNode>(legalMoves.Length);
                // for (var i = 0; i < legalMoves.Length; i++)
                // {
                //     var move = legalMoves[i];
                //     sim.Node.Children[move] = new MctsNode
                //     {
                //         Parent = sim.Node,
                //         Prior = sim.Peval[move],
                //         MoveFromParent = move,
                //     };
                // }
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
        }
    }
}
