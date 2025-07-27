using System.Diagnostics;
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

internal record Experience(Tensor PrevState, Tensor State, float Reward);

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
            var old = _buffer.Dequeue();
            old.PrevState.Dispose();
            old.State.Dispose();
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
        int experienceBufferSize = 64)
    {
        _device = cuda.is_available() ? CUDA : CPU;
        _valueNet = new SmallNetwork().to(_device);
        _valueNet.train();
        _targetNet = new SmallNetwork().to(_device);
        _targetNet.load_state_dict(_valueNet.state_dict());
        _targetNet.eval();
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
            resultingStates.Add(Board2Tensor(internalGame));
            game.Undo();
        }
        using var stateBatch = cat(resultingStates.ToArray()).to(_device);
        using var values = _valueNet.Forward(stateBatch).squeeze();

        // greedily pick the highest value next state
        var bestIdx = values.argmax().ToInt32();
        return legalMoves[bestIdx];
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

        for (int episode = 0; episode < nEpisodes; ++episode)
        {
            var game = CodingAdventureChessGame.StandardGame();
            var players = new Dictionary<Color, IChessAgent>
            {
                { Color.White, this },
                { Color.Black, opponent }
            };
            var prevState = Board2Tensor(game);
            var episodeLosses = new List<float>();
            var episodeReward = 0.0f;

            while (!game.IsGameOver())
            {
                var move = players[game.Turn()].NextMove(game, TimeSpan.FromMilliseconds(turnTimeLimitMs));
                game.MakeMove(move);
                var state = Board2Tensor(game);
                var reward = Reward(game);

                // Assume this agent is always White for now
                if (game.Turn() != Color.White)
                {
                    _replayBuffer.Push(new Experience(prevState, state, reward));
                    episodeReward += reward;
                }

                if (game.Turn() == Color.White)
                {
                    prevState = state;
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

            // Print progress every 10 episodes
            if ((episode + 1) % 10 == 0)
            {
                var recentCount = Math.Min(100, _episodeWins.Count);
                var recentWins = _episodeWins.Skip(Math.Max(0, _episodeWins.Count - recentCount)).ToList();
                float winRate = recentWins.Count > 0 ? recentWins.Sum() / (float)recentWins.Count : 0f;
                Console.WriteLine($"Ep {episode + 1}/{nEpisodes} | WinRate: {winRate:F3} | AvgLoss: {avgLoss:F4} | Reward: {episodeReward:F2}");
            }

            // Print detailed stats every print_every episodes
            int printEvery = 100;
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
    }

    private static Tensor Board2Tensor(CodingAdventureChessGame game)
    {
        var tensor = zeros(8, 8, 8, float32);
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
            tensor[pieceIdx][row][col] = (float)color;
        }
        var fullmove = game.FullmoveCount();
        if (fullmove > 0)
        {
            tensor[6].fill_(1.0f / fullmove);
        }
        if (game.Turn() == Color.White)
        {
            tensor[6][0].fill_(1.0f);
        }
        else
        {
            tensor[6][0].fill_(-1.0f);
        }
        tensor[7].fill_(1.0f);
        return tensor;
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
        var prevStates = batch.Select(x => x.PrevState).ToArray();
        var states = batch.Select(x => x.State).ToArray();
        var rewards = batch.Select(x => x.Reward).ToArray();
        using var statesT = cat(prevStates).to(_device);
        using var nextStatesT = cat(states).to(_device);
        using var rewardsT = tensor(rewards, dtype: ScalarType.Float32).to(_device);

        _valueNet.train();
        using var currentValues = _valueNet.Forward(statesT).squeeze();

        Tensor targetValues;
        using (no_grad())
        {
            _targetNet.eval();
            using var nextValues = _targetNet.Forward(nextStatesT).squeeze();
            targetValues = rewardsT + _gamma * nextValues;
        }

        using var loss = nn.functional.mse_loss(currentValues, targetValues);
        _optimizer.zero_grad();
        loss.backward();
        nn.utils.clip_grad_norm_(_valueNet.parameters(), _maxGradNorm);
        _optimizer.step();
        UpdateTargetNetwork();
        targetValues.Dispose();
        return loss.ToSingle();
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
