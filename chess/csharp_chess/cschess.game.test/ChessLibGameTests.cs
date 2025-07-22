using Shouldly;

namespace cschess.game.test;

public class ChessLibGameTests
{
    [Fact]
    public void PlaysFullRandomGame()
    {
        // var random = new Random();
        // var game = ChessLibChessGame.StandardGame();
        //
        // var numHalfMoves = 0;
        //
        // while (!game.IsGameOver())
        // {
        //     var move = random.Choice(game.LegalMoves());
        //     game.MakeMove(move);
        //
        //     (numHalfMoves++).ShouldBeLessThan(500);
        // }
    }
}

// public static class RandomExtensions
// {
//     public static T Choice<T>(this Random random, IEnumerable<T> source)
//     {
//         var sourceList = source.ToList();
//         var index = random.Next(sourceList.Count);
//         return sourceList[index];
//     }
// }
