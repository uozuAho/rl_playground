using cschess.agents;

namespace cschess.tournament.test;

public class TournamentTests
{
    [Fact]
    public void Play2Randoms()
    {
        var results = Tournament.RunWith(
            new TournamentOptions(2, TimeSpan.FromMilliseconds(1)),
            new TournamentEntrant(new RandomAgent(), "random1"),
            new TournamentEntrant(new RandomAgent(), "random2")
        );
    }
}
