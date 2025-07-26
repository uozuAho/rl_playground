using cschess.agents;
using cschess.tournament;

var results = Tournament.RunWith(
    new TournamentOptions(
        NumGamesPerMatch: 3,
        TurnTimeLimit: TimeSpan.FromMilliseconds(10)),
    new TournamentEntrant(new RandomAgent(), "random"),
    new TournamentEntrant(new CodingAdventureAgent(), "CodingAdventure")
);

Console.WriteLine();
Console.WriteLine("Results:");
Console.WriteLine();

foreach (var match in results.Matches)
{
    Console.WriteLine($"{match.White.Name} (white) vs {match.Black.Name} (black)");

    var numGames = match.Games.Count;
    var avgLen = match.Games.Sum(x => x.Halfmoves) / numGames;
    var whiteWins = match.Games.Count(x => x.WhiteWon);
    var draws = match.Games.Count(x => x.IsDraw);
    var blackWins = numGames - whiteWins - draws;
    Console.WriteLine($"{whiteWins}/{draws}/{blackWins}. Avg halfmoves: {avgLen}");

    // todo: visualise results like
    // X vs Y: win/draw/loss  WWWWWWWWWWWWWWWWWW..............LLLLLLLLLLLLLLLLL

    // todo: ranking? ELO? total wins/loss

    // todo: end types
    // var asdf  = match.Games
    //     .GroupBy(x => x.FinalState)
    //     .OrderDescending(g => g.)
}
