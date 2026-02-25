using System.Diagnostics;
using cschess.game;

namespace cschess.agents;

using Value = double;
using MoveProbs = Dictionary<Move, double>;

public delegate IEnumerable<(MoveProbs, Value)> BatchEvalFn(IEnumerable<IChessGame> games);

public interface IEvaluator
{
    IEnumerable<(MoveProbs, Value)> BatchEval(IEnumerable<IChessGame> games);
}

public record MctsNode
{
    public MctsNode? Parent { get; init; }
    public double Prior { get; init; }
    public Move? MoveFromParent { get; init; }
    public Dictionary<Move, MctsNode> Children = new Dictionary<Move, MctsNode>();
    public int Visits = 0;
    public double TotalValue = 0.0;
    public double? VEst = null;

    internal IChessGame? _state = null;

    public IChessGame State()
    {
        if (_state == null)
        {
            Debug.Assert(Parent != null);
            _state = Parent.State().Copy();
            Debug.Assert(MoveFromParent != null);
            _state.MakeMove(MoveFromParent);
        }

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

    internal bool IsExpanded => Children.Count > 0;

    internal bool IsTerminal => State().IsGameOver();
}

internal class MctsSimState
{
    public MctsSimState(MctsNode root)
    {
        Root = root;
        Node = root;
    }

    internal MctsNode Root { get; set; }
    internal MctsNode Node { get; set; }
    internal double? TerminalValue = null;
    internal Dictionary<Move, double>? Peval = null;
    internal double? Veval = null;

    internal void Reset()
    {
        Node = Root;
        TerminalValue = null;
        Peval = null;
        Veval = null;
    }
}

public class ParallelMcts
{
    private List<IChessGame> States;
    private IEvaluator Evaluator;
    private int NumSimulations;
    private double CPuct;
    private bool AddDirichletNoise;
    private double DirichletAlpha;
    private double DirichletEpsilon;

    private int _simCount = 0;
    private List<MctsSimState> _sims = new();

    public ParallelMcts(
        List<IChessGame> states,
        IEvaluator evaluator,
        int numSimulations,
        double cPuct = 1.0,
        bool addDirichletNoise = false,
        double dirichletAlpha = 0.3,
        double dirichletEpsilon = 0.25
    )
    {
        States = states;
        Evaluator = evaluator;
        NumSimulations = numSimulations;
        CPuct = cPuct;
        AddDirichletNoise = addDirichletNoise;
        DirichletAlpha = dirichletAlpha;
        DirichletEpsilon = dirichletEpsilon;
    }

    public List<MctsNode> Run()
    {
        _sims = States
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

        while (_simCount < NumSimulations)
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

            // Selection: traverse tree using PUCT until we reach a leaf
            while (sim.Node.IsExpanded && !sim.Node.IsTerminal)
            {
                sim.Node = sim.Node.Children.Values.MaxBy(c => c.Puct(CPuct))!;
            }

            if (sim.Node.IsTerminal)
            {
                var gameState = sim.Node.State().GameState();
                var turn = sim.Node.State().Turn();
                var otherPlayer = turn == Color.White ? Color.Black : Color.White;

                if (gameState.IsDraw)
                    sim.TerminalValue = 0.0;
                else if (
                    (gameState.IsWhiteWin && otherPlayer == Color.White)
                    || (gameState.IsBlackWin && otherPlayer == Color.Black)
                )
                    sim.TerminalValue = 1.0;
                else
                    sim.TerminalValue = -1.0;
            }
        }
    }

    private void Eval()
    {
        var envs = _sims.Select(s => s.Node.State());
        var pvs = Evaluator.BatchEval(envs).ToList();
        for (int i = 0; i < _sims.Count; i++)
        {
            var (p, v) = pvs[i];
            _sims[i].Peval = p;
            _sims[i].Veval = v;
        }
    }

    private void FinishSim()
    {
        foreach (var sim in _sims)
        {
            Debug.Assert(sim.Veval != null);
            Debug.Assert(sim.Peval != null);

            if (sim.TerminalValue == null)
            {
                // evaluate gives the value for the current player, we want
                // for the previous player - just need to invert the value
                sim.Veval = -sim.Veval;
                sim.Node.VEst = sim.Veval;

                if (ReferenceEquals(sim.Node, sim.Root) && AddDirichletNoise)
                {
                    AddDirichletNoiseToEval(sim);
                }

                // Expand: create child nodes for all valid actions
                foreach (var action in sim.Node.State().LegalMoves())
                {
                    sim.Node.Children[action] = new MctsNode
                    {
                        Parent = sim.Node,
                        Prior = sim.Peval[action],
                        MoveFromParent = action,
                    };
                }
            }

            var value =
                (sim.TerminalValue.HasValue && sim.TerminalValue.Value != 0.0)
                    ? sim.TerminalValue.Value
                    : sim.Veval!.Value;

            // Backpropagation: update values up the search path
            var node = (MctsNode?)sim.Node;
            while (node != null)
            {
                node.Visits++;
                node.TotalValue += value;
                node = node.Parent;
                value = -value;
            }
        }
    }

    private void AddDirichletNoiseToEval(MctsSimState sim)
    {
        throw new NotImplementedException("Dirichlet noise not yet implemented");
    }
}
