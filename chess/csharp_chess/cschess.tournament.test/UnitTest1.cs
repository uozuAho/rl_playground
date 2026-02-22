using cschess.agents;

namespace cschess.tournament.test;

public class UnitTest1
{
    [Fact]
    public void Test1()
    {
        var results = Tournament.RunWith(
            new TournamentOptions(2, TimeSpan.FromMilliseconds(1)),
            new TournamentEntrant(new RandomAgent(), "random1"),
            new TournamentEntrant(new RandomAgent(), "random2")
        );
    }
}
