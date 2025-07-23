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

            (numHalfMoves++).ShouldBeLessThan(500);
        }

        numHalfMoves.ShouldBeGreaterThan(50);
    }
}
