using cschess.agents.AlphaZero;
using Shouldly;

namespace cschess.agents.test.AlphaZero;

public class PlayerTests
{
    [Fact]
    public void asdf()
    {
        var oneGame = Player.SelfPlayGames(new UniformEvaluator(), nGames: 1, nMctsSims: 3, cPuct: 1.0, temperature: 1.0,
            dirichletAlpha: 0.1, dirichletEpsilon: 0.01).ToList();

        oneGame.Count.ShouldBeGreaterThan(0);
    }
}
