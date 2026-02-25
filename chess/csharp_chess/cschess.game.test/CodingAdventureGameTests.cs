using cschess.csutils;
using Shouldly;

namespace cschess.game.test;

public class CodingAdventureGameTests
{
    [Fact]
    public void PlaysFullRandomGame()
    {
        var random = new Random();
        var game = CodingAdventureChessGame.StandardGame();

        var numHalfMoves = 0;

        while (!game.IsGameOver())
        {
            var move = random.Choice(game.LegalMoves());
            game.MakeMove(move);

            (numHalfMoves++).ShouldBeLessThan(2000);
        }

        numHalfMoves.ShouldBeGreaterThan(2);
    }

    [Fact]
    public void UndoWorks()
    {
        var random = new Random();
        var game = CodingAdventureChessGame.StandardGame();

        while (!game.IsGameOver())
        {
            var prevFen = game.Fen();
            var move = random.Choice(game.LegalMoves());
            game.MakeMove(move);
            var currentFen = game.Fen();
            game.Undo();

            game.Fen().ShouldBe(prevFen);
            game.MakeMove(move);
            game.Fen().ShouldBe(currentFen);
        }
    }
}
