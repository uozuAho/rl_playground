using cschess.game;
using TorchSharp;
using static TorchSharp.torch;

namespace cschess.agents.AlphaZero;

public class Learner
{
    public static (float policyLoss, float valueLoss) UpdateNet(
        IAzNet net,
        optim.Optimizer optimizer,
        ICodec codec, // todo: rename these to input/output codecs? or merge them
        IEnumerable<GameSample> samples,
        bool maskInvalidActions
    )
    {
        var states = new List<IChessGame>();
        var probs = new List<Dictionary<Move, float>>();
        var values = new List<float>();

        foreach (var sample in samples)
        {
            states.Add(sample.Game);
            probs.Add(sample.MoveProbs);
            values.Add(sample.FinalReward);
        }

        var encStates = codec.StatesToNumbers(states);
        var encProbs = codec.ProbsToNumbers(probs, codec);
        var encValues = values.ToArray();

        var tStates = from_array(encStates);
        var tProbs = from_array(encProbs);
        var tVals = from_array(encValues);

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
