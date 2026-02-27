using System.Diagnostics;
using cschess.game;
using MoreLinq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace cschess.agents.AlphaZero;

/// <summary>
/// An NN that alphazero can use. Given a chess game, outputs move probabilities
/// and value estimate.
/// </summary>
public interface IAzNet
{
    ICodec Codec { get; }
    IEnumerable<(Dictionary<Move, float>, float)> BatchEval(IEnumerable<IChessGame> games);
}

public class ResNet : IAzNet
{
    public ICodec Codec { get; } = new Codec4096();
    private readonly ResNetModule _model;
    private readonly Device _device;

    public ResNet(int numResBlocks, int numHidden, Device device)
    {
        _model = new ResNetModule(numResBlocks, numHidden, Codec).to(device);
        _device = device;
    }

    public IEnumerable<(Dictionary<Move, float>, float)> BatchEval(IEnumerable<IChessGame> games)
    {
        var gameList = games.ToList();
        var pvs = Pv(gameList).ToList();
        Debug.Assert(pvs.Count == gameList.Count);
        foreach (var pvgs in pvs.Zip(gameList))
        {
            var (pv, game) = pvgs;
            var (p, v) = pv;
            yield return (Codec.Probdist2Dict(p, game), v);
        }
    }

    /// <summary>
    /// games to list(probs, value) output
    /// </summary>
    private IEnumerable<(float[], float)> Pv(IEnumerable<IChessGame> games)
    {
        var (logits, values) = Forward(games);
        var parr = logits.softmax(dim: 1).cpu().data<float>().ToArray();
        var varr = values.squeeze().cpu().data<float>().ToArray();
        Debug.Assert(parr.Length == varr.Length * Codec.ActionSize);
        // todo: check batch is in the right axis. assert probdist
        return parr.Batch(Codec.ActionSize).Zip(varr);
    }

    /// <summary>
    /// games to raw model tensor outputs
    /// </summary>
    private (Tensor, Tensor) Forward(IEnumerable<IChessGame> games)
    {
        var arr = StatesToNumbers(games);
        var input = from_array(arr).to(_device);
        return _model.forward(input);
    }

    private static int PieceLayer(PieceType piece)
    {
        return piece switch
        {
            PieceType.Pawn => 0,
            PieceType.Rook => 1,
            PieceType.Knight => 2,
            PieceType.Bishop => 3,
            PieceType.Queen => 4,
            PieceType.King => 5,
            _ => throw new ArgumentOutOfRangeException(nameof(piece), piece, null),
        };
    }

    private static float[,,,] StatesToNumbers(IEnumerable<IChessGame> games)
    {
        var gamesList = games.ToList();
        var batch = new float[gamesList.Count, 8, 8, 8];
        for (var b = 0; b < gamesList.Count; b++)
        {
            var arr = StateToNumbers(gamesList[b]);
            for (var i = 0; i < 8; i++)
            for (var j = 0; j < 8; j++)
            for (var k = 0; k < 8; k++)
                batch[b, i, j, k] = arr[i, j, k];
        }
        return batch;
    }

    private static float[,,] StateToNumbers(IChessGame game)
    {
        var state = new float[8, 8, 8];

        for (var rank = 0; rank < 8; rank++)
        {
            for (var file = 0; file < 8; file++)
            {
                var square = Square.FromRankAndFile(rank, file);
                var piece = game.PieceAt(square);
                if (piece == null)
                    continue;

                var sign = game.ColorAt(square) == Color.White ? 1f : -1f;

                var layer = PieceLayer(piece.Value);
                state[layer, rank, file] = sign;
            }
        }

        var moveValue = 1f / game.FullmoveCount();
        for (var r = 0; r < 8; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                state[6, r, c] = moveValue;
            }
        }

        var turnValue = game.Turn() == Color.White ? 1f : -1f;
        for (var c = 0; c < 8; c++)
        {
            state[6, 0, c] = turnValue;
        }

        for (var r = 0; r < 8; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                state[7, r, c] = 1f;
            }
        }

        return state;
    }
}

internal sealed class ResNetModule : nn.Module<Tensor, (Tensor, Tensor)>
{
    private readonly Sequential _start;
    private readonly Sequential _backbone;
    private readonly Sequential _policyHead;
    private readonly Sequential _valueHead;

    public ResNetModule(int numResBlocks, int numHidden, ICodec codec)
        : base("ResNet")
    {
        _start = nn.Sequential(
            ("conv1", nn.Conv2d(8, numHidden, kernel_size: 3, padding: 1)),
            ("bn1", nn.BatchNorm2d(numHidden)),
            ("relu", nn.ReLU())
        );
        _backbone = nn.Sequential(
            Enumerable.Range(0, numResBlocks).Select(i => new ResBlock(numHidden))
        );
        _policyHead = nn.Sequential(
            nn.Conv2d(numHidden, 32, kernel_size: 3, padding: 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            // 8x8 for chess board
            nn.Linear(32 * 8 * 8, codec.ActionSize)
        );
        _valueHead = nn.Sequential(
            nn.Conv2d(numHidden, 3, kernel_size: 3, padding: 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 1),
            nn.Tanh()
        );
    }

    public override (Tensor, Tensor) forward(Tensor x)
    {
        x = _start.forward(x);
        x = _backbone.forward(x);
        var policy = _policyHead.forward(x);
        var value = _valueHead.forward(x);
        return (policy, value);
    }
}

internal sealed class ResBlock : nn.Module<Tensor, Tensor>
{
    private readonly Conv2d _conv1;
    private readonly Conv2d _conv2;
    private readonly BatchNorm2d _bn1;
    private readonly BatchNorm2d _bn2;

    public ResBlock(int numHidden)
        : base("ResBlock")
    {
        _conv1 = nn.Conv2d(numHidden, numHidden, kernel_size: 3, padding: 1);
        _bn1 = nn.BatchNorm2d(numHidden);
        _conv2 = nn.Conv2d(numHidden, numHidden, kernel_size: 3, padding: 1);
        _bn2 = nn.BatchNorm2d(numHidden);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var residual = x;
        x = nn.functional.relu(_bn1.forward(_conv1.forward(x)));
        x = _bn2.forward(_conv2.forward(x));
        x += residual;
        x = nn.functional.relu(x);
        return x;
    }
}
