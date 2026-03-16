using cschess.agents.AlphaZero;
using cschess.game;
using Shouldly;
using static TorchSharp.torch;

namespace cschess.agents.test;

public class AzSelfPlayerTests
{
    [Fact]
    public void runs_and_stops()
    {
        var device = CPU;
        var net = new ResNet(1, 1, device);
        using var tmcts = new AzSelfPlayer(net, 1, 1, device);
        var inGame = CodingAdventureChessGame.StandardGame();
        var inFen = inGame.Fen();
        tmcts.Enqueue(inGame);
        tmcts.Start();
        tmcts.StopAndWait();
        var outGame = tmcts.DoneQueue.Take();

        outGame.IsGameOver().ShouldBeTrue();
        inGame.Fen().ShouldBe(inFen, "should not modify input game");
    }
}
