using cschess.agents;
using cschess.agents.AlphaZero;
using cschess.game;
using static TorchSharp.torch;

runs_and_stops();
return;

void runs_and_stops()
{
    var net = new ResNet(1,1,CPU);
    using var tmcts = new AzSelfPlayer(net, 1, 1);
    var inGame = CodingAdventureChessGame.StandardGame();
    var inFen = inGame.Fen();
    tmcts.Enqueue(inGame);
    tmcts.Start();
    tmcts.StopAndWait();
    var outGame = tmcts.DoneQueue.Take();
}
