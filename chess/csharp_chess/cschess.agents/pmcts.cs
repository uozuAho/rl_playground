using System.Diagnostics;
using cschess.game;

namespace cschess.agents;

using MoveProbs = Dictionary<Move, double>;

public interface IEvaluator
{
    IEnumerable<(MoveProbs, double)> BatchEval(IEnumerable<IChessGame> games);
}

public record MctsNode
{
    public MctsNode? Parent { get; init; }
    public double Prior { get; init; }
    public Move? MoveFromParent { get; init; }
    public readonly Dictionary<Move, MctsNode> Children = new();
    public int Visits;
    public double TotalValue;

    internal IChessGame? _state;

    public IChessGame State()
    {
        if (_state != null)
            return _state;

        Debug.Assert(Parent != null);
        _state = Parent.State().Copy();
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

    internal bool IsExpanded => Children.Count > 0;

    internal bool IsTerminal => State().IsGameOver();
}

internal class MctsSimState(MctsNode root)
{
    internal MctsNode Root { get; set; } = root;
    internal MctsNode Node { get; set; } = root;
    internal double? TerminalValue = null;
    internal MoveProbs? Peval = null;
    internal double? Veval = null;

    internal void Reset()
    {
        Node = Root;
        TerminalValue = null;
        Peval = null;
        Veval = null;
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
    private double _dirichletAlpha = dirichletAlpha;
    private double _dirichletEpsilon = dirichletEpsilon;

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

            while (sim.Node is { IsExpanded: true, IsTerminal: false })
            {
                sim.Node = sim.Node.Children.Values.MaxBy(c => c.Puct(cPuct))!;
            }

            if (sim.Node.IsTerminal)
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
        }
    }

    private void Eval()
    {
        var envs = _sims.Select(s => s.Node.State());
        var pvs = evaluator.BatchEval(envs).ToList();
        for (var i = 0; i < _sims.Count; i++)
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
                sim.Veval = -sim.Veval;

                if (ReferenceEquals(sim.Node, sim.Root) && addDirichletNoise)
                {
                    AddDirichletNoiseToEval(sim);
                }

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

            var node = sim.Node;
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
