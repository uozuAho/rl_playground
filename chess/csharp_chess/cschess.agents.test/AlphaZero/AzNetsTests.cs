using cschess.agents.AlphaZero;
using cschess.csutils;
using cschess.game;
using Shouldly;
using static TorchSharp.torch;

namespace cschess.agents.test.AlphaZero;

public class AzNetsTests
{
    [Fact]
    public void BatchEval_doesnt_throw()
    {
        var net = new ResNet(1, 1, CPU);
        var game = CodingAdventureChessGame.StandardGame();
        using (no_grad())
        {
            var pvs = net.BatchEval([game]).ToList();
            pvs.Count.ShouldBe(1);
            var (mp, v) = pvs[0];
            var probs = mp.Values.ToList();
            Maths.IsProbDist(probs).ShouldBe(true);
        }
    }
}
