using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch;
using cschess.game;
using TorchSharp.Modules;

namespace cschess.agents;

public sealed class SmallNetwork : nn.Module
{
    private readonly Sequential _seq;

    public SmallNetwork()
        : base("SmallNetwork")
    {
        // Sequential is good for memory, see: https://github.com/dotnet/TorchSharp/blob/main/docfx/articles/memory.md
        _seq = nn.Sequential(
            ("conv1", nn.Conv2d(8, 32, 3, stride: 1, padding: 1)),
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2d(32, 64, 3, stride: 1, padding: 1)),
            ("relu2", nn.ReLU()),
            ("conv3", nn.Conv2d(64, 32, 3, stride: 1, padding: 1)),
            ("relu3", nn.ReLU()),
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(32 * 8 * 8, 128)),
            ("relu4", nn.ReLU()),
            ("dropout", nn.Dropout(0.3)),
            ("fc2", nn.Linear(128, 64)),
            ("relu5", nn.ReLU()),
            ("fc3", nn.Linear(64, 1)),
            ("tanh", nn.Tanh())
        );

        RegisterComponents();
    }

    public Tensor Forward(Tensor x)
    {
        return _seq.forward(x);
    }
}

/// <summary>
///
/// </summary>
/// <param name="PrevState"></param>
/// <param name="State">If null, prevstate is an endstate</param>
/// <param name="Reward"></param>
internal record Experience(float[,,] PrevState, float[,,]? State, float Reward);

class ExperienceReplay
{
    private readonly Queue<Experience> _buffer;
    private readonly int _capacity;
    private readonly Random _random = new();

    public ExperienceReplay(int capacity)
    {
        _capacity = capacity;
        _buffer = new Queue<Experience>(capacity);
    }

    public void Push(Experience item)
    {
        if (_buffer.Count >= _capacity)
        {
            _buffer.Dequeue();
        }
        _buffer.Enqueue(item);
    }

    public List<Experience> Sample(int n)
    {
        return _buffer.OrderBy(_ => _random.Next()).Take(Math.Min(n, _buffer.Count)).ToList();
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
    private readonly ExperienceReplay _replayBuffer;
    private readonly float _maxGradNorm = 1.0f;
    private readonly Device _device;

    // Training metrics
    private readonly List<int> _episodeWins = [];
    private readonly List<int> _episodeHalfmoves = [];
    private readonly List<float> _episodeLosses = [];
    private readonly List<float> _episodeRewards = [];

    public GreedyNnAgent(
        float lr = 1e-3f,
        float gamma = 0.99f,
        float tau = 0.001f,
        int batchSize = 32,
        int experienceBufferSize = 64,
        string device = "cpu")
    {
        _device = device switch
        {
            "cpu" => CPU,
            "gpu" => CUDA,
            _ => throw new ArgumentOutOfRangeException(nameof(device), device, null)
        };
        _valueNet = new SmallNetwork().to(_device);
        _valueNet.train();
        _targetNet = new SmallNetwork().to(_device);
        _targetNet.load_state_dict(_valueNet.state_dict());
        _targetNet.eval();
        foreach (var parameter in _targetNet.parameters())
        {
            parameter.requires_grad = false;
        }
        _optimizer = optim.Adam(_valueNet.parameters(), lr: lr);
        _gamma = gamma;
        _tau = tau;
        _batchSize = batchSize;
        _replayBuffer = new ExperienceReplay(experienceBufferSize);
    }

    public Move NextMove(IChessGame game, TimeSpan timeout)
    {
        if (game is not CodingAdventureChessGame internalGame)
            throw new InvalidOperationException("Only CodingAdventureChessGame is supported");

        var legalMoves = game.LegalMoves().ToList();
        if (legalMoves.Count == 0)
            throw new InvalidOperationException("No legal moves available");

        // get the value of all next states
        var resultingStates = new List<Tensor>();
        foreach (var move in legalMoves)
        {
            game.MakeMove(move);
            resultingStates.Add(tensor(Board2Array(internalGame)));
            game.Undo();
        }
        using var stateBatch = stack(resultingStates.ToArray()).to(_device);
        using (no_grad())
        {
            _valueNet.eval();
            using var values = _valueNet.Forward(stateBatch).squeeze();

            // greedily pick the highest value next state
            var bestIdx = values.argmax().ToInt32();
            return legalMoves[bestIdx];
        }
    }

    private float Reward(CodingAdventureChessGame state)
    {
        var gameState = state.GameState();
        // assumes greedy agent is white
        if (gameState.IsWhiteWin) { return 1.0f; }
        if (gameState.IsBlackWin) { return -1.0f; }

        return 0.0f;
    }

    public void TrainAgainst(
        IChessAgent opponent,
        int nEpisodes,
        int? halfmoveLimit = null,
        int turnTimeLimitMs = 10)
    {
        if (halfmoveLimit != null)
            throw new NotImplementedException("halfmoveLimit not implemented");

        var totalTrainingTimer = Stopwatch.StartNew();

        for (var episode = 0; episode < nEpisodes; ++episode)
        {
            Console.WriteLine(".");
            var game = CodingAdventureChessGame.StandardGame();
            var players = new Dictionary<Color, IChessAgent>
            {
                { Color.White, this },
                { Color.Black, opponent }
            };
            var prevState = Board2Array(game);
            var episodeLosses = new List<float>();
            var episodeReward = 0.0f;

            while (!game.IsGameOver())
            {
                var move = players[game.Turn()].NextMove(game, TimeSpan.FromMilliseconds(turnTimeLimitMs));
                game.MakeMove(move);
                var state = Board2Array(game);
                var reward = Reward(game);
                episodeReward += reward;

                if (game.Turn() != Color.White)
                {
                    // White just moved.
                    // Assume this agent is always White for now.
                    // Only store experience for white moves.
                    _replayBuffer.Push(new Experience(prevState, state, reward));
                }

                if (game.Turn() == Color.White)
                {
                    prevState = state;
                    if (game.IsGameOver())
                    {
                        _replayBuffer.Push(new Experience(state, null, reward));
                    }
                }

                var loss = TrainStep();
                if (loss.HasValue)
                    episodeLosses.Add(loss.Value);
            }

            // Track metrics for this episode
            var gameState = game.GameState();
            _episodeWins.Add(gameState.IsWhiteWin ? 1 : 0);
            _episodeHalfmoves.Add(game.HalfmoveCount());
            var avgLoss = episodeLosses.Count > 0 ? episodeLosses.Average() : 0.0f;
            _episodeLosses.Add(avgLoss);
            _episodeRewards.Add(episodeReward);

            // Print progress every episode
            if ((episode + 1) % 1 == 0)
            {
                var recentCount = Math.Min(100, _episodeWins.Count);
                var recentWins = _episodeWins.Skip(Math.Max(0, _episodeWins.Count - recentCount)).ToList();
                float winRate = recentWins.Count > 0 ? recentWins.Sum() / (float)recentWins.Count : 0f;
                Console.WriteLine($"Ep {episode + 1}/{nEpisodes} | WinRate: {winRate:F3} | AvgLoss: {avgLoss:F4} | Reward: {episodeReward:F2}");
            }

            // Print detailed stats every print_every 10 episodes
            int printEvery = 10;
            if ((episode + 1) % printEvery == 0)
            {
                var stats = GetTrainingStats();
                Console.WriteLine($"\nEpisode {episode + 1}/{nEpisodes}");
                Console.WriteLine($"  Win Rate (recent): {stats["recent_win_rate"]:F3}");
                Console.WriteLine($"  Win Rate (overall): {stats["overall_win_rate"]:F3}");
                Console.WriteLine($"  Avg Game Length: {stats["recent_avg_game_length"]:F1}");
                Console.WriteLine($"  Avg Loss: {stats["recent_avg_loss"]:F4}");
                Console.WriteLine($"  Avg Reward: {stats["recent_avg_reward"]:F3}");
                Console.WriteLine($"  Buffer Size: {_replayBuffer.Count}");
            }
        }
        totalTrainingTimer.Stop();

        Console.WriteLine($"Trained {nEpisodes} episodes ({_episodeHalfmoves.Sum()} total states) " +
                          $"in {totalTrainingTimer.Elapsed}");
        var gameRate = nEpisodes / (float)totalTrainingTimer.Elapsed.TotalSeconds;
        var stepRate = _episodeHalfmoves.Sum() / totalTrainingTimer.Elapsed.TotalSeconds;
        Console.WriteLine($"{gameRate:F2} games/sec, {stepRate:F2} steps/sec");
    }

    private static float[,,] Board2Array(CodingAdventureChessGame game)
    {
        var data = new float[8, 8, 8];

        for (var i = 0; i < 64; ++i)
        {
            var row = i / 8;
            var col = i % 8;
            var pieceType = game.PieceAt(i);
            if (!pieceType.HasValue)
                continue;
            var color = game.ColorAt(i);
            Debug.Assert(color != Color.None);
            var pieceIdx = (int)pieceType - 1;
            data[pieceIdx, row, col] = (float)color;
        }

        var fullmove = game.FullmoveCount();
        if (fullmove > 0)
        {
            for (var row = 0; row < 8; ++row)
            for (var col = 0; col < 8; ++col)
                data[6, row, col] = 1.0f / fullmove;
        }
        var turnVal = game.Turn() == Color.White ? 1.0f : -1.0f;
        for (var col = 0; col < 8; ++col)
            data[6, 0, col] = turnVal;

        for (var row = 0; row < 8; ++row)
        for (var col = 0; col < 8; ++col)
            data[7, row, col] = 1.0f;

        // var tensor = torch.tensor(data, dtype: ScalarType.Float32);
        return data;
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

    private float? TrainStep()
    {
        if (_replayBuffer.Count < _batchSize)
            return null;

        var batch = _replayBuffer.Sample(_batchSize);
        var nonEndExperiences = batch.Where(x => x.State is not null).ToArray();
        var endExperiences = batch.Where(x => x.State is null).ToArray();

        var prevStates = nonEndExperiences.Select(x => tensor(x.PrevState)).ToArray();
        var states = nonEndExperiences.Select(x => tensor(x.State!)).ToArray();
        var rewards = nonEndExperiences.Select(x => x.Reward).ToArray();
        using var prevStatesT = stack(prevStates).to(_device);
        using var statesT = stack(states).to(_device);
        using var rewardsT = tensor(rewards, dtype: ScalarType.Float32).to(_device);

        _valueNet.train();
        using var currentValues = _valueNet.Forward(prevStatesT).squeeze();

        Tensor targetValues;
        using (no_grad())
        {
            _targetNet.eval();
            using var nextValues = _targetNet.Forward(statesT).squeeze();
            targetValues = rewardsT + _gamma * nextValues;
        }

        using var loss = nn.functional.mse_loss(currentValues, targetValues);
        _optimizer.zero_grad();
        loss.backward();
        nn.utils.clip_grad_norm_(_valueNet.parameters(), _maxGradNorm);
        _optimizer.step();
        UpdateTargetNetwork();
        targetValues.Dispose();
        var nonEndLoss = loss.ToSingle();

        if (endExperiences.Length != 0)
        {
            var endStates = endExperiences.Select(x => tensor(x.PrevState)).ToArray();
            var endRewards = endExperiences.Select(x => x.Reward).ToArray();
            using var endStatesT = stack(endStates).to(_device);
            using var endTargetsT = tensor(endRewards).unsqueeze(1).to(_device);

            using var endEstimates = _valueNet.Forward(endStatesT);

            using var lossEnd = nn.functional.mse_loss(endEstimates, endTargetsT);
            _optimizer.zero_grad();
            lossEnd.backward();
            nn.utils.clip_grad_norm_(_valueNet.parameters(), _maxGradNorm);
            _optimizer.step();
            UpdateTargetNetwork();
        }

        return nonEndLoss;
    }

    private Dictionary<string, double> GetTrainingStats()
    {
        if (_episodeWins.Count == 0)
            return new Dictionary<string, double>();

        var recentEpisodes = Math.Min(100, _episodeWins.Count);
        var recentWins = _episodeWins.Skip(Math.Max(0, _episodeWins.Count - recentEpisodes)).ToList();
        var recentLengths = _episodeHalfmoves.Skip(Math.Max(0, _episodeHalfmoves.Count - recentEpisodes)).ToList();
        var recentLosses = _episodeLosses.Skip(Math.Max(0, _episodeLosses.Count - recentEpisodes)).ToList();
        var recentRewards = _episodeRewards.Skip(Math.Max(0, _episodeRewards.Count - recentEpisodes)).ToList();

        return new Dictionary<string, double>
        {
            ["total_episodes"] = _episodeWins.Count,
            ["recent_win_rate"] = recentWins.Count > 0 ? recentWins.Sum() / (float)recentWins.Count : 0,
            ["recent_avg_game_length"] = recentLengths.Count > 0 ? recentLengths.Average() : 0.0,
            ["recent_avg_loss"] = recentLosses.Count > 0 ? recentLosses.Average() : 0,
            ["recent_avg_reward"] = recentRewards.Count > 0 ? recentRewards.Average() : 0,
            ["overall_win_rate"] = _episodeWins.Count > 0 ? _episodeWins.Sum() / (float)_episodeWins.Count : 0
        };
    }
}
