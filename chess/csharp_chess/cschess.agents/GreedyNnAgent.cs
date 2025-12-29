using System.Diagnostics;
using cschess.csutils;
using cschess.game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

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

    public GreedyNnAgent(
        float lr = 1e-3f,
        float gamma = 0.99f,
        float tau = 0.001f,
        int batchSize = 32,
        int experienceBufferSize = 64,
        string device = "cpu"
    )
    {
        _device = device switch
        {
            "cpu" => CPU,
            "gpu" => CUDA,
            _ => throw new ArgumentOutOfRangeException(nameof(device), device, null),
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
        var stateBatch = stack(resultingStates.ToArray()).to(_device);
        using (no_grad())
        {
            _valueNet.eval();
            var values = _valueNet.Forward(stateBatch).squeeze();

            // greedily pick the highest value next state
            var bestIdx = values.argmax().ToInt32();
            return legalMoves[bestIdx];
        }
    }

    private float Reward(CodingAdventureChessGame state)
    {
        var gameState = state.GameState();
        // assumes greedy agent is white
        if (gameState.IsWhiteWin)
        {
            return 1.0f;
        }
        if (gameState.IsBlackWin)
        {
            return -1.0f;
        }

        return 0.0f;
    }

    /// <summary>
    /// Train against the given opponents. Opponents are chosen randomly each episode.
    /// </summary>
    public void TrainAgainst(
        IChessAgent[] opponents,
        int nEpisodes,
        int? halfmoveLimit = null,
        int turnTimeLimitMs = 10,
        Action<List<EpisodeStats>>? epCallback = null
    )
    {
        var random = new Random();
        var epStats = new List<EpisodeStats>();
        var totalTrainingTimer = Stopwatch.StartNew();
        var epTime = TimeSpan.Zero;

        var d = NewDisposeScope();

        for (var episode = 0; episode < nEpisodes; ++episode)
        {
            var start = totalTrainingTimer.Elapsed;

            var game = CodingAdventureChessGame.StandardGame();
            var opponent = random.Choice(opponents);
            var players = new Dictionary<Color, IChessAgent>
            {
                { Color.White, this },
                { Color.Black, opponent },
            };
            var prevState = Board2Array(game);
            var episodeLosses = new List<float>();
            var episodeReward = 0.0f;

            d.Dispose();
            d = NewDisposeScope();

            while (!game.IsGameOver() && game.HalfmoveCount() < (halfmoveLimit ?? int.MaxValue))
            {
                if ((game.HalfmoveCount() + 1) % 100 == 0)
                {
                    // prevent GPU OOM during long episodes
                    d.Dispose();
                    d = NewDisposeScope();
                }

                var move = players[game.Turn()]
                    .NextMove(game, TimeSpan.FromMilliseconds(turnTimeLimitMs));
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
            var end = totalTrainingTimer.Elapsed;
            epTime = end - start;

            var gameState = game.GameState();
            epStats.Add(new EpisodeStats
            {
                Win = gameState.IsWhiteWin,
                Draw = gameState.IsDraw || gameState.IsInProgress,  // in progress means halfmove limit
                Loss = gameState.IsBlackWin,
                Reward = episodeReward,
                Halfmoves = game.HalfmoveCount(),
                AvgLoss = episodeLosses.Count > 0 ? episodeLosses.Average() : 0,
                Duration = epTime
            });

            epCallback?.Invoke(epStats);

            if ((episode + 1) % 1 == 0)
            {
                PrintSummary(nEpisodes, epStats, episode);
            }
        }

        d.Dispose();

        totalTrainingTimer.Stop();

        var wins = epStats.Sum(x => x.Win ? 1 : 0);
        var losses = epStats.Sum(x => x.Loss ? 1 : 0);
        var draws = epStats.Sum(x => x.Draw ? 1 : 0);
        var epRate = nEpisodes / totalTrainingTimer.Elapsed.TotalSeconds;
        var posRate = epStats.Sum(x => x.Halfmoves) / totalTrainingTimer.Elapsed.TotalSeconds;
        Console.WriteLine($"Trained {nEpisodes} eps in {totalTrainingTimer.Elapsed} ({epRate:F2} games/s, {posRate:F2} pos/s)");
        Console.WriteLine($"Wins: {wins}");
        Console.WriteLine($"Draws: {draws}");
        Console.WriteLine($"Losses: {losses}");
    }

    private static void PrintSummary(int nEpisodes, List<EpisodeStats> epStats, int episode)
    {
        const int recentWindowSize = 5;
        var wins = epStats.Sum(x => x.Win ? 1 : 0);
        var losses = epStats.Sum(x => x.Loss ? 1 : 0);
        var draws = epStats.Sum(x => x.Draw ? 1 : 0);
        var avgHalfmoves = epStats.TakeLast(recentWindowSize).Average(x => x.Halfmoves);
        var avgLoss = epStats.TakeLast(recentWindowSize).Average(x => x.AvgLoss);
        var avgDuration = epStats.TakeLast(recentWindowSize).Average(x => x.Duration.TotalSeconds);
        var epRate = 1 / avgDuration;
        var posRate = avgHalfmoves / avgDuration;
        Console.WriteLine(
            $"Ep {episode + 1}/{nEpisodes} | W/D/L: {wins}/{draws}/{losses} | AvgLoss: {avgLoss:F4} | {epRate:F2} games/s, {posRate:F2} pos/s"
        );
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
        var prevStatesT = stack(prevStates).to(_device);
        var statesT = stack(states).to(_device);
        var rewardsT = tensor(rewards, dtype: ScalarType.Float32).to(_device);

        _valueNet.train();
        var currentValues = _valueNet.Forward(prevStatesT).squeeze();

        Tensor targetValues;
        using (no_grad())
        {
            _targetNet.eval();
            var nextValues = _targetNet.Forward(statesT).squeeze();
            targetValues = rewardsT + _gamma * nextValues;
        }

        var loss = nn.functional.mse_loss(currentValues, targetValues);
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
            var endStatesT = stack(endStates).to(_device);
            var endTargetsT = tensor(endRewards).unsqueeze(1).to(_device);

            var endEstimates = _valueNet.Forward(endStatesT);

            var lossEnd = nn.functional.mse_loss(endEstimates, endTargetsT);
            _optimizer.zero_grad();
            lossEnd.backward();
            nn.utils.clip_grad_norm_(_valueNet.parameters(), _maxGradNorm);
            _optimizer.step();
            UpdateTargetNetwork();
        }

        return nonEndLoss;
    }
}

public record EpisodeStats
{
    public bool Win { get; init; }
    public bool Draw { get; init; }
    public bool Loss { get; init; }
    public int Halfmoves { get; init; }
    public float AvgLoss { get; init; }
    public float Reward { get; init; }
    public TimeSpan Duration { get; init; }
}
