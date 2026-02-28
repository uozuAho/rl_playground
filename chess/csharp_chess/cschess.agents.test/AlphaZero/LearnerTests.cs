using cschess.agents.AlphaZero;
using cschess.csutils;
using cschess.game;
using Shouldly;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace cschess.agents.test.AlphaZero;

public class LearnerTests
{
    [Fact]
    public void LearnOneStep()
    {
        var net = new ResNet(1, 1, CPU);
        var optimiser = new Adam(net.ModelParams());
        var gameSamples = new[]
        {
            new GameSample(
                CodingAdventureChessGame.StandardGame(),
                new Dictionary<Move, float>
                {
                    { Move.FromUci("e2e3"), 1.0f },
                },
                1.0f
            ),
        };
        net.Train();
        var (ploss, vloss) = Learner.UpdateNet(
            net,
            optimiser,
            new Codec4096(),
            gameSamples,
            false
        );
        ploss.ShouldNotBe(float.NaN);
        vloss.ShouldNotBe(float.NaN);
        net.Eval();
        using (no_grad())
        {
            var eval = net.BatchEval([CodingAdventureChessGame.StandardGame()]).ToList();
            eval.Count.ShouldBe(1);
            Maths.IsProbDist(eval[0].Item1.Values).ShouldBeTrue();
        }
    }
}
