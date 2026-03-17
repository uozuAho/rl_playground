using cschess.agents.AlphaZero;
using cschess.game;
using static TorchSharp.torch;

runs_and_stops();
return;

void runs_and_stops()
{
    var device = CUDA;
    const int mctsSims = 10;
    const int batchSize = 10;
    const int numGames = 10;
    var net = new ResNet(2, 48, device);
    const LogLevel logLevel = LogLevel.Debug;
    using var tmcts = new AzSelfPlayer(net, mctsSims, batchSize, device, logLevel: logLevel);
    for (var i = 0; i < numGames; i++)
    {
        tmcts.Enqueue(CodingAdventureChessGame.StandardGame());
    }
    tmcts.Start();
    tmcts.StopAndWait();
}
