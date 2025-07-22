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

public static class RandomExtensions
{
    public static T Choice<T>(this Random random, IEnumerable<T> source)
    {
        var sourceList = source.ToList();
        var index = random.Next(sourceList.Count);
        return sourceList[index];
    }
}
