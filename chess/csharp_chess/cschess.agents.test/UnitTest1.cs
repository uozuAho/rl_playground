using cschess.game;
using Shouldly;

namespace cschess.agents.test;

public class UnitTest1
{
    [Fact]
    public void MakesMove()
    {
        var game = CodingAdventureChessGame.StandardGame();
        var adventureAgent = new CodingAdventureAgent();

        var move = adventureAgent.NextMove(game);

        move.ShouldNotBeNull();
    }
}
