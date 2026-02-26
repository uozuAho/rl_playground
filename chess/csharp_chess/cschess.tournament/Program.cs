using cschess.agents;
using cschess.tournament;

var results = Tournament.RunWith(
    new TournamentOptions(NumGamesPerMatch: 3, TurnTimeLimit: TimeSpan.FromMilliseconds(10)),
    new TournamentEntrant(new RandomAgent(), "random"),
    new TournamentEntrant(new CodingAdventureAgent(), "CodingAdventure")

    // slow!
    // new TournamentEntrant(MctsAgent.RandomRollout(10), "MctsRR-10")
);

Console.WriteLine();
Console.WriteLine("Results:");
Console.WriteLine();

foreach (var match in results.Matches)
{
    Console.WriteLine(match.Summary());
}

Console.WriteLine();
results.PrintStats();
