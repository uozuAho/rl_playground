using cschess.agents.AlphaZero;
using cschess.game;
using static TorchSharp.torch;

runs_and_stops();
return;

void runs_and_stops()
{
    var device = CPU;
    const int mctsSims = 1;
    const int batchSize = 1;
    const int unbatchSize = 1;
    const int numGames = 1;
    var net = new ResNet(1, 1, device);
    const LogLevel logLevel = LogLevel.Debug;
    using var tmcts = new AzSelfPlayer2(
        net,
        mctsSims,
        batchSize,
        unbatchSize,
        device,
        logLevel: logLevel
    );
    for (var i = 0; i < numGames; i++)
    {
        tmcts.Enqueue(CodingAdventureChessGame.StandardGame());
    }
    tmcts.Start();
    tmcts.StopAndWait();
}
