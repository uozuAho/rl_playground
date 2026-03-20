using cschess.agents.AlphaZero;
using cschess.game;
using Shouldly;
using static TorchSharp.torch;

namespace cschess.agents.test;

public class AzSelfPlayer2Tests
{
    [Fact]
    public void runs_and_stops()
    {
        var device = CPU;
        var net = new ResNet(1, 1, device);
        const int mctsSims = 1;
        const int batchSize = 1;
        const int unbatchSize = 1;
        using var tmcts = new AzSelfPlayer2(
            net,
            mctsSims,
            batchSize,
            unbatchSize,
            device,
            logLevel: LogLevel.None
        );
        var inGame = CodingAdventureChessGame.StandardGame();
        var inFen = inGame.Fen();
        tmcts.Enqueue(inGame);
        tmcts.Start();
        tmcts.StopAndWait();
        var outGame = tmcts.DoneQueue.Take();

        outGame.IsGameOver().ShouldBeTrue();
        inGame.Fen().ShouldBe(inFen, "should not modify input game");
    }

    [Fact]
    public void SelfPlayGame_mcts1_start()
    {
        // todo: extract these classes/make them private
        var initGame = CodingAdventureChessGame.StandardGame();
        var initLegalMoves = initGame.LegalMoves().ToList();
        var initFen = initGame.Fen();
        var root = new MctsNode3
        {
            State = initGame
        };
        const int nMctsSims = 1;
        const double cPuct = 1.0;
        var game = new SelfPlayGame(root, nMctsSims, cPuct, addDirichletNoise: true, 0.3, 0.25);

        game.SearchRoot.Children.ShouldBeNull();
        game.Advance(); // expands to leaf, ready for eval
        game.SearchRoot.Children.ShouldNotBeNull();
        game.SearchRoot.Children.Count.ShouldBe(initLegalMoves.Count);

        Should.Throw<Exception>(() => game.Advance(), "advance expects eval to be done");

        game.Peval = initLegalMoves.ToDictionary(m => m, _ => 0.1f);
        game.Veval = 1.0f;

        game.Advance(); // num sims is one, so should make move
        game.RootFen.ShouldNotBe(initFen);
    }

    [Fact]
    public void SelfPlayGame_mcts10_start()
    {
        // todo: extract these classes/make them private
        var initGame = CodingAdventureChessGame.StandardGame();
        var initLegalMoves = initGame.LegalMoves().ToList();
        var initFen = initGame.Fen();
        var root = new MctsNode3
        {
            State = initGame
        };
        const int nMctsSims = 10;
        const double cPuct = 1.0;
        var game = new SelfPlayGame(root, nMctsSims, cPuct, addDirichletNoise: true, 0.3, 0.25);

        game.SearchRoot.Children.ShouldBeNull();
        game.Advance(); // expands to leaf, ready for eval
        game.SearchRoot.Children.ShouldNotBeNull();
        game.SearchRoot.Children.Count.ShouldBe(initLegalMoves.Count);

        Should.Throw<Exception>(() => game.Advance(), "advance expects eval to be done");

        game.Peval = initLegalMoves.ToDictionary(m => m, _ => 0.1f);
        game.Veval = 1.0f;

        game.Advance();
        game.RootFen.ShouldBe(initFen); // < 10 sims, should not make move
        for (var i = 0; i < 9; i++)
        {
            game.Peval = game.SearchNode.State!.LegalMoves().ToDictionary(m => m, _ => 0.1f);
            game.Veval = 1.0f;
            game.Advance();
        }
        game.RootFen.ShouldNotBe(initFen);
    }
}
