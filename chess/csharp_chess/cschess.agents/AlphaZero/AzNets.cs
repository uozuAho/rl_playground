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
    Device Device { get; }
    ICodec Codec { get; }
    IEnumerable<(Dictionary<Move, float>, float)> BatchEval(IEnumerable<IChessGame> games);

    // Raw tensor passthrough for learning
    (Tensor policy, Tensor value) Forward(Tensor states);
    public IEnumerable<Parameter> ModelParams();
}

public class ResNet : IAzNet, IEvaluator
{
    public ICodec Codec { get; } = new Codec4096();
    private readonly ResNetModule _model;
    public Device Device { get; private set; }

    public ResNet(int numResBlocks, int numHidden, Device device)
    {
        _model = new ResNetModule(numResBlocks, numHidden, Codec, device).to(device);
        Device = device;
    }

    public void Train()
    {
        _model.train();
    }

    public void Eval()
    {
        _model.eval();
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

    public (Tensor policy, Tensor value) Forward(Tensor states)
    {
        return _model.forward(states);
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
        return parr.Batch(Codec.ActionSize).Zip(varr);
    }

    /// <summary>
    /// games to raw model tensor outputs
    /// </summary>
    private (Tensor, Tensor) Forward(IEnumerable<IChessGame> games)
    {
        var arr = Codec.States2Array(games);
        var input = from_array(arr).to(Device);
        return _model.forward(input);
    }

    public IEnumerable<Parameter> ModelParams()
    {
        return _model.parameters();
    }
}

internal sealed class ResNetModule : nn.Module<Tensor, (Tensor, Tensor)>
{
    private readonly Sequential _start;
    private readonly Sequential _backbone;
    private readonly Sequential _policyHead;
    private readonly Sequential _valueHead;

    public ResNetModule(int numResBlocks, int numHidden, ICodec codec, Device device)
        : base("ResNet")
    {
        _start = nn.Sequential(
                ("conv1", nn.Conv2d(8, numHidden, kernel_size: 3, padding: 1)),
                ("bn1", nn.BatchNorm2d(numHidden)),
                ("relu", nn.ReLU())
            )
            .to(device);
        _backbone = nn.Sequential(
                Enumerable.Range(0, numResBlocks).Select(i => new ResBlock(numHidden))
            )
            .to(device);
        _policyHead = nn.Sequential(
                nn.Conv2d(numHidden, 32, kernel_size: 3, padding: 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten(),
                // 8x8 for chess board
                nn.Linear(32 * 8 * 8, codec.ActionSize)
            )
            .to(device);
        _valueHead = nn.Sequential(
                nn.Conv2d(numHidden, 3, kernel_size: 3, padding: 1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3 * 8 * 8, 1),
                nn.Tanh()
            )
            .to(device);

        RegisterComponents();
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
