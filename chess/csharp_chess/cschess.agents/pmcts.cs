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
    public double Prior { get; init; }
    public Move? MoveFromParent { get; init; }
    public readonly Dictionary<Move, MctsNode> Children = new();
    public int Visits;
    public double TotalValue;

    // todo: make this private
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
        var games = _sims.Where(s => !s.Node.IsTerminal).Select(s => s.Node.State()).ToList();
        if (games.Count == 0)
        {
            return;
        }
        var pvs = evaluator.BatchEval(games).ToList();
        foreach (var spv in _sims.Where(s => !s.Node.IsTerminal).Zip(pvs))
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
                    sim.Peval = Maths.AddDirichletNoise(
                        sim.Peval,
                        dirichletAlpha,
                        dirichletEpsilon
                    );
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

            var value = sim.TerminalValue ?? sim.Veval!.Value;

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
}
