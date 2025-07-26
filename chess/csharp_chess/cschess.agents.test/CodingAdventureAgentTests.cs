using cschess.game;
using Shouldly;

namespace cschess.agents.test;

public class CodingAdventureAgentTests
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
