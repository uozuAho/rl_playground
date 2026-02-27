using cschess.agents.AlphaZero;
using cschess.game;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace cschess.agents.test.AlphaZero;

public class LearnerTests
{
    [Fact]
    public void asdf()
    {
        var net = new ResNet(1, 1, CPU);
        var optimiser = new Adam(net.ModelParams());
        var gameSamples = new[]
        {
            new GameSample(
                CodingAdventureChessGame.StandardGame(),
                new Dictionary<Move, float>
                {
                    // todo: Move("e2e3")
                    // todo: throw on bad rank/file
                    {
                        new Move(Square.FromRankAndFile(2, 8), Square.FromRankAndFile(3, 8)),
                        1.0f
                    },
                },
                1.0f
            ),
        };
        net.Train();
        for (var i = 0; i < 100; i++)
        {
            var (ploss, vloss) = Learner.UpdateNet(
                net,
                optimiser,
                new ResNetEncoder(),
                new Codec4096(),
                gameSamples,
                false
            );
        }
        net.Eval();
        using (no_grad())
        {
            var eval = net.BatchEval([CodingAdventureChessGame.StandardGame()]).ToList();
            Console.WriteLine("asdF");
        }
        // todo: assert loss aint nan
    }
}
