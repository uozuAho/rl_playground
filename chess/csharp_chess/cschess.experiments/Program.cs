using cschess.agents.AlphaZero;
using cschess.game;
using static TorchSharp.torch;

runs_and_stops();
return;

void runs_and_stops()
{
    var device = CUDA;
    const int mctsSims = 20;
    const int batchSize = 40;
    const int numGames = 40;
    var net = new ResNet(2, 48, device);
    using var tmcts = new AzSelfPlayer(net, mctsSims, batchSize, device, logLevel: LogLevel.Info);
    for (var i = 0; i < numGames; i++)
    {
        tmcts.Enqueue(CodingAdventureChessGame.StandardGame());
    }
    tmcts.Start();
    tmcts.StopAndWait();
}
