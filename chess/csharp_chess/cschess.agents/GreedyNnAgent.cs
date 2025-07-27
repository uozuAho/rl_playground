using TorchSharp;
using static TorchSharp.torch;
using cschess.game;
using TorchSharp.Modules;

namespace cschess.agents;

public sealed class SmallNetwork : nn.Module
{
    private nn.Module _conv1,
        _conv2,
        _conv3;
    private nn.Module _fc1,
        _fc2,
        _fc3;
    private nn.Module _dropout;

    public SmallNetwork()
        : base("SmallNetwork")
    {
        _conv1 = nn.Conv2d(8, 32, 3, stride: 1, padding: 1);
        _conv2 = nn.Conv2d(32, 64, 3, stride: 1, padding: 1);
        _conv3 = nn.Conv2d(64, 32, 3, stride: 1, padding: 1);
        _fc1 = nn.Linear(32 * 8 * 8, 128);
        _fc2 = nn.Linear(128, 64);
        _fc3 = nn.Linear(64, 1);
        _dropout = nn.Dropout(0.3);

        RegisterComponents();
    }

    public Tensor Forward(Tensor x)
    {
        // todo: use Sequential instead of this mess. Not sure how to do the view at x4.
        // Sequential is good for memory, see: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/memory.md
        using var x1 = nn.functional.relu(((Conv2d)_conv1).forward(x));
        using var x2 = nn.functional.relu(((Conv2d)_conv2).forward(x1));
        using var x3 = nn.functional.relu(((Conv2d)_conv3).forward(x2));
        using var x4 = x3.view(x3.shape[0], -1);
        using var x5 = nn.functional.relu(((Linear)_fc1).forward(x4));
        using var x6 = ((Dropout)_dropout).forward(x5);
        using var x7 = nn.functional.relu(((Linear)_fc2).forward(x6));
        var x8 = nn.functional.tanh(((Linear)_fc3).forward(x7));
        return x8;
    }
}

public class ExperienceReplay<T>
{
    private readonly Queue<T> _buffer;
    private readonly int _capacity;
    private readonly Random _random = new();

    public ExperienceReplay(int capacity = 10000)
    {
        _capacity = capacity;
        _buffer = new Queue<T>(capacity);
    }

    public void Push(T item)
    {
        if (_buffer.Count >= _capacity)
            _buffer.Dequeue();
        _buffer.Enqueue(item);
    }

    public List<T> Sample(int batchSize)
    {
        return _buffer.OrderBy(x => _random.Next()).Take(Math.Min(batchSize, _buffer.Count)).ToList();
    }

    public int Count => _buffer.Count;
}

public class GreedyNnAgent : IChessAgent
{
    private readonly SmallNetwork _valueNet;
    private readonly SmallNetwork _targetNet;
    private readonly optim.Optimizer _optimizer;
    private readonly float _gamma;
    private readonly float _tau;
    private readonly int _batchSize;
    private readonly ExperienceReplay<(Tensor state, Tensor nextState, float reward)> _replayBuffer;
    private readonly float _maxGradNorm = 1.0f;
    private readonly Device _device;

    // Training metrics
    private readonly List<int> _episodeWins = new();
    private readonly List<int> _episodeGameLengths = new();
    private readonly List<float> _episodeLosses = new();
    private readonly List<float> _episodeRewards = new();
    private int _episodeCount = 0;

    public GreedyNnAgent(float lr = 1e-3f, float gamma = 0.99f, float tau = 0.001f, int batchSize = 32)
    {
        _device = cuda.is_available() ? CUDA : CPU;
        _valueNet = new SmallNetwork().to(_device);
        _targetNet = new SmallNetwork().to(_device);
        _targetNet.load_state_dict(_valueNet.state_dict());
        _optimizer = optim.Adam(_valueNet.parameters(), lr: lr);
        _gamma = gamma;
        _tau = tau;
        _batchSize = batchSize;
        _replayBuffer = new ExperienceReplay<(Tensor, Tensor, float)>();
    }

    public Move NextMove(IChessGame game, TimeSpan timeout)
    {
        throw new NotImplementedException();
        // todo: implement next move

        // var legalMoves = game.LegalMoves().ToList();
        // if (legalMoves.Count == 0)
        //     throw new InvalidOperationException("No legal moves available");
        //
        // var resultingStates = new List<Tensor>();
        // foreach (var move in legalMoves)
        // {
        //     game.Step(move);
        //     resultingStates.Add(StateToTensor(game.StateNP()));
        //     game.Undo();
        // }
        // using var stateBatch = cat(resultingStates.ToArray(), 0).to(_device);
        // using var values = _smallNet.Forward(stateBatch).squeeze();
        // var bestIdx = values.argmax().ToInt32();
        // return legalMoves[bestIdx];
    }

    public void TrainAgainst(
        IChessAgent opponent,
        int nEpisodes,
        int? halfmoveLimit = null)
    {
        for (int i = 0; i < nEpisodes; ++i)
        {
            var game = CodingAdventureChessGame.StandardGame();
            var done = false;
        }
    }

    private Tensor StateToTensor(float[,,] stateArray)
    {
        var t = torch.tensor(stateArray, dtype: ScalarType.Float32).unsqueeze(0).to(_device);
        return t;
    }

    private void UpdateTargetNetwork()
    {
        var targetParams = _targetNet.parameters().ToArray();
        var valueParams = _valueNet.parameters().ToArray();
        for (int i = 0; i < targetParams.Length; i++)
        {
            targetParams[i].copy_(_tau * valueParams[i] + (1.0f - _tau) * targetParams[i]);
        }
    }

    private void AddExperience(float[,,] state, float[,,] nextState, float reward)
    {
        _replayBuffer.Push((StateToTensor(state), StateToTensor(nextState), reward));
    }

    private float? TrainStep()
    {
        if (_replayBuffer.Count < _batchSize)
            return null;
        var batch = _replayBuffer.Sample(_batchSize);
        var states = batch.Select(x => x.state).ToArray();
        var nextStates = batch.Select(x => x.nextState).ToArray();
        var rewards = batch.Select(x => x.reward).ToArray();
        using var statesT = cat(states, 0).to(_device);
        using var nextStatesT = cat(nextStates, 0).to(_device);
        using var rewardsT = torch.tensor(rewards, dtype: ScalarType.Float32).to(_device);
        using var currentValues = _valueNet.Forward(statesT).squeeze();
        using var nextValues = _targetNet.Forward(nextStatesT).squeeze();
        using var targetValues = rewardsT + _gamma * nextValues;
        using var loss = nn.functional.mse_loss(currentValues, targetValues);
        _optimizer.zero_grad();
        loss.backward();
        nn.utils.clip_grad_norm_(_valueNet.parameters(), _maxGradNorm);
        _optimizer.step();
        UpdateTargetNetwork();
        return loss.ToSingle();
    }
}
