using cschess.agents.AlphaZero;
using cschess.game;
using static TorchSharp.torch;

runs_and_stops();
// asdf();
return;

void runs_and_stops()
{
    var device = CUDA;
    const int mctsSims = 60;
    const int batchSize = 10;
    const int unbatchSize = 10;
    const int numGames = 20;
    var net = new ResNet(2, 48, device);
    const LogLevel logLevel = LogLevel.Info;
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

void asdf()
{
    ExperimentSaturateGpu.EvaluateSaturateGpu();
}
