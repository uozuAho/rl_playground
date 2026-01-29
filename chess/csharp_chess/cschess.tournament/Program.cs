using cschess.agents;
using cschess.tournament;

var results = Tournament.RunWith(
    new TournamentOptions(NumGamesPerMatch: 3, TurnTimeLimit: TimeSpan.FromMilliseconds(10)),
    new TournamentEntrant(new RandomAgent(), "random"),
    new TournamentEntrant(new CodingAdventureAgent(), "CodingAdventure")
);

Console.WriteLine();
Console.WriteLine("Results:");
Console.WriteLine();

foreach (var match in results.Matches)
{
    Console.WriteLine(match.Summary());

    // todo: visualise results like
    // X vs Y: win/draw/loss  WWWWWWWWWWWWWWWWWW..............LLLLLLLLLLLLLLLLL

    // todo: end types
    // var asdf  = match.Games
    //     .GroupBy(x => x.FinalState)
    //     .OrderDescending(g => g.)
}
