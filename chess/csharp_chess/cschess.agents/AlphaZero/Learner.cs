using cschess.game;
using TorchSharp;
using static TorchSharp.torch;

namespace cschess.agents.AlphaZero;

public class Learner
{
    public static (float policyLoss, float valueLoss) UpdateNet(
        IAzNet net,
        optim.Optimizer optimizer,
        IEnumerable<GameSample> samples,
        bool maskInvalidActions
    )
    {
        var codec = net.Codec;
        var states = new List<IChessGame>();
        var probs = new List<Dictionary<Move, float>>();
        var values = new List<float>();

        foreach (var sample in samples)
        {
            states.Add(sample.Game);
            probs.Add(sample.MoveProbs);
            values.Add(sample.FinalReward);
        }

        var encStates = codec.States2Array(states);
        var encProbs = codec.Probs2Array(probs);
        var encValues = codec.Values2Array(values);

        var tStates = from_array(encStates).to(net.Device);
        var tProbs = from_array(encProbs).to(net.Device);
        var tVals = from_array(encValues).to(net.Device);

        var (outpol, outval) = net.Forward(tStates);

        // todo: mask invalid actions

        var ploss = nn.functional.cross_entropy(outpol, tProbs);
        var vloss = nn.functional.mse_loss(outval, tVals);
        var loss = ploss + vloss;

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        return (ploss.ToSingle(), vloss.ToSingle());
    }
}
