using TorchSharp.Modules;

namespace cschess.agents.AlphaZero;

using static TorchSharp.torch;

public class Az
{
    public static void Train()
    {
        const int numGames = 1;
        var net = new ResNet(1, 1, CPU);
        var optimiser = new Adam(net.ModelParams());

        for (var i = 0; i < numGames; i++)
        {
            IEnumerable<GameSample> samples;
            using (no_grad())
            {
                samples = Player.SelfPlayGames(
                    net,
                    nGames: 1,
                    nMctsSims: 10,
                    cPuct: 2.0,
                    temperature: 1.25,
                    dirichletAlpha: 0.3,
                    dirichletEpsilon: 0.25
                ).ToList();
            }
            // todo: remove net.codec
            var (ploss, vloss) = Learner.UpdateNet(net, optimiser, net.Codec, samples, false);
            Console.WriteLine($"{ploss}, {vloss}");
        }
    }
}
