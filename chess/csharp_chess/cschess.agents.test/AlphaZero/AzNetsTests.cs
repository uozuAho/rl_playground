using cschess.agents.AlphaZero;
using cschess.csutils;
using cschess.game;
using Shouldly;
using static TorchSharp.torch;

namespace cschess.agents.test.AlphaZero;

public class AzNetsTests
{
    [Fact]
    public void BatchEval()
    {
        var net = new ResNet(1, 1, CPU);
        var games = new[]
        {
            CodingAdventureChessGame.StandardGame(),
            CodingAdventureChessGame.StandardGame(),
        };
        using (no_grad())
        {
            var pvs = net.BatchEval(games).ToList();
            pvs.Count.ShouldBe(games.Length);
            foreach (var (mp, v) in pvs)
            {
                var probs = mp.Values.ToList();
                Maths.IsProbDist(probs).ShouldBe(true);
                v.ShouldNotBe(float.NaN);
            }
        }
    }
}
