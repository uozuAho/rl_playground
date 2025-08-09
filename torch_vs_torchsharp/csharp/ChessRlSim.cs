using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace csharp;

public static class ChessRlSim
{
    public static void Run(string[] args)
    {
        var device = args.Contains("cpu") ? "cpu" : args.Contains("gpu") ? "gpu" : "both";

        if (device is "cpu" or "both")
        {
            Console.WriteLine("cpu:");
            var agent = new GreedyChessAgent(device: "cpu");
            var opponent = new RandomAgent();
            agent.TrainAgainst(opponent, 100);
        }

        if (device is "gpu" or "both")
        {
            Console.WriteLine();
            Console.WriteLine("gpu:");
            var agent = new GreedyChessAgent(device: "cuda");
            var opponent = new RandomAgent();
            agent.TrainAgainst(opponent, 2000);
        }
    }
}

public class FakeChessGame
{
    public const int White = 1;
    public const int Black = -1;

    public int turn = White;
    private bool _done = false, _donePrev = false;
    private int? _winner = null, _winnerPrev = null;
    private int _totalHalfmoves = 0;
    private float[,,] _state;
    private float[,,] _statePrev;
    private static Random rng = new Random();

    public FakeChessGame()
    {
        _state = RandomState();
        _statePrev = _state;
    }

    public List<int> LegalMoves()
    {
        int n = rng.Next(1, 21);
        var moves = new List<int>();
        for (int i = 0; i < n; i++)
            moves.Add(rng.Next(0, 4097));
        return moves;
    }

    public int? Winner() => _winner;

    public (bool, int) Step(int move)
    {
        _totalHalfmoves++;
        _statePrev = _state;
        _state = RandomState();
        _donePrev = _done;
        _done = _totalHalfmoves > 50;
        if (_done)
        {
            _winnerPrev = _winner;
            int[] choices = { 0, -1, 1 };
            _winner = choices[rng.Next(choices.Length)];
        }
        return (_done, rng.Next(-1, 1));
    }

    public void Undo()
    {
        _done = _donePrev;
        _winner = _winnerPrev;
        _totalHalfmoves--;
        _state = _statePrev;
    }

    public float[,,] AsFloatArray() => _state;

    private float[,,] RandomState()
    {
        var arr = new float[8, 8, 8];
        for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
        for (int k = 0; k < 8; k++)
            arr[i, j, k] = (float)rng.NextDouble();
        return arr;
    }
}

public interface IChessAgent
{
    int GetAction(FakeChessGame game);
}

public class RandomAgent : IChessAgent
{
    private static readonly Random Rng = new();

    public int GetAction(FakeChessGame game)
    {
        var moves = game.LegalMoves();
        return moves[Rng.Next(moves.Count)];
    }
}

public sealed class ValueNetwork : nn.Module
{
    private readonly Sequential _seq;

    public ValueNetwork(string name = "ValueNetwork") : base(name)
    {
        _seq = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding: 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding: 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding: 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        );
        RegisterComponents();
    }

    public Tensor Forward(Tensor x)
    {
        return _seq.forward(x);
    }
}

public class ExperienceReplay
{
    private readonly int _capacity;
    private readonly Queue<(float[,,], float[,,], float)> _buffer;

    public ExperienceReplay(int capacity = 128)
    {
        _capacity = capacity;
        _buffer = new Queue<(float[,,], float[,,], float)>();
    }

    public void Push(float[,,] state, float[,,] nextState, float reward)
    {
        if (_buffer.Count >= _capacity)
            _buffer.Dequeue();
        _buffer.Enqueue((state, nextState, reward));
    }

    public List<(float[,,], float[,,], float)> Sample(int batchSize)
    {
        var list = _buffer.ToList();
        var rng = new Random();
        return list.OrderBy(_ => rng.Next()).Take(Math.Min(batchSize, list.Count)).ToList();
    }

    public int Count => _buffer.Count;
}

public class GreedyChessAgent : IChessAgent
{
    private readonly int _player;
    private readonly ValueNetwork _valueNet;
    private readonly ValueNetwork _targetNet;
    private readonly optim.Optimizer _optimizer;
    private readonly float _gamma;
    private readonly float _tau;
    private readonly int _batchSize;
    private readonly ExperienceReplay _replayBuffer;
    private const float MaxGradNorm = 1.0f;
    private readonly Device _device;

    public GreedyChessAgent(
        int player = FakeChessGame.White,
        float lr = 1e-3f,
        float gamma = 0.99f,
        float tau = 0.001f,
        int batchSize = 32,
        string device = "cpu"
    )
    {
        _player = player;
        _device = torch.device(device);
        _valueNet = new ValueNetwork().to(_device);
        _targetNet = new ValueNetwork().to(_device);
        _targetNet.load_state_dict(_valueNet.state_dict());
        _optimizer = optim.Adam(_valueNet.parameters(), lr: lr);
        _gamma = gamma;
        _tau = tau;
        _batchSize = batchSize;
        _replayBuffer = new ExperienceReplay();
    }

    public int GetAction(FakeChessGame env)
    {
        if (_player != FakeChessGame.White) throw new Exception("Only WHITE supported");
        if (env.turn != _player) throw new Exception("Not this agent's turn");
        var legalMoves = env.LegalMoves();
        if (legalMoves.Count == 0) throw new Exception("No legal moves available");
        var resultingStates = new List<float[,,]>();
        foreach (var move in legalMoves)
        {
            env.Step(move);
            resultingStates.Add(env.AsFloatArray());
            env.Undo();
        }

        var stateTensors = stack(resultingStates
                .Select(arr => tensor(arr, dtype: ScalarType.Float32))
                .ToArray())
            .to(_device);
        using (no_grad())
        {
            var values = _valueNet.Forward(stateTensors).squeeze();
            var bestIdx = values.argmax().ToInt32();
            return legalMoves[bestIdx];
        }
    }

    private void UpdateTargetNetwork()
    {
        var targetParams = _targetNet.parameters().ToArray();
        var valueParams = _valueNet.parameters().ToArray();
        for (int i = 0; i < targetParams.Length; i++)
        {
            targetParams[i].mul_(1 - _tau).add_(valueParams[i].mul(_tau));
        }
    }

    public void TrainAgainst(IChessAgent opponent, int nEpisodes)
    {
        var sw = Stopwatch.StartNew();
        var totalSteps = 0;
        var numEpsPerDispose = 100;

        var d = NewDisposeScope();

        for (var ep = 0; ep < nEpisodes; ep++)
        {
            if (ep > 0 && ep % numEpsPerDispose == 0)
            {
                d.Dispose();
                d = NewDisposeScope();
            }

            var game = new FakeChessGame();
            var done = false;
            var players = new Dictionary<int, IChessAgent>
            {
                { FakeChessGame.White, this },
                { FakeChessGame.Black, opponent }
            };
            var prevState = game.AsFloatArray();

            while (!done)
            {
                var playerAgent = players[game.turn];
                var move = playerAgent.GetAction(game);
                (done, var reward) = game.Step(move);
                totalSteps++;
                var state = game.AsFloatArray();
                if (game.turn != _player)
                {
                    AddExperience(prevState, state, reward);
                }
                if (game.turn == _player)
                    prevState = state;
                TrainStep();
            }
        }
        sw.Stop();
        var duration = sw.Elapsed.TotalSeconds;
        var epRate = nEpisodes / duration;
        var posRate = totalSteps / duration;
        Console.WriteLine($"Done training in {duration:F2}s. {epRate:F2} eps/sec, {posRate:F2} moves/sec");
    }

    private void TrainStep()
    {
        if (_replayBuffer.Count < _batchSize) return;
        var batch = _replayBuffer.Sample(_batchSize);
        var states = batch.Select(x => tensor(x.Item1, dtype: ScalarType.Float32)).ToArray();
        var nextStates = batch.Select(x => tensor(x.Item2, dtype: ScalarType.Float32)).ToArray();
        var rewards = batch.Select(x => x.Item3).ToArray();
        var statesT = stack(states).to(_device);
        var nextStatesT = stack(nextStates).to(_device);
        var rewardsT = tensor(rewards, dtype: ScalarType.Float32).to(_device);
        var currentValues = _valueNet.Forward(statesT).squeeze();
        using (no_grad())
        {
            var nextValues = _targetNet.Forward(nextStatesT).squeeze();
            var targetValues = rewardsT + _gamma * nextValues;
            var loss = nn.functional.mse_loss(currentValues, targetValues);
            _optimizer.zero_grad();
            loss.backward();
            nn.utils.clip_grad_norm_(_valueNet.parameters(), MaxGradNorm);
            _optimizer.step();
            UpdateTargetNetwork();
        }
    }

    private void AddExperience(float[,,] state, float[,,] nextState, float reward)
    {
        _replayBuffer.Push(state, nextState, reward);
    }
}
