using cschess.agents;
using cschess.tournament;

// train value network
// using cschess.experiments;
// Console.WriteLine(string.Join(",", args));
// var fenScoresPath = args[0];
// ValueNetworkTrainer.TrainAndTestValueNetwork(fenScoresPath);

var nnAgent = new GreedyNnAgent();
var randomAgent = new RandomAgent();

MatchResult PlayMatch(IChessAgent white, string whiteName, IChessAgent black, string blackName)
{
    var tournamentOptions = new TournamentOptions(NumGamesPerMatch: 5, TurnTimeLimit: TimeSpan.FromMilliseconds(10));
    return Tournament.PlaySingleMatch(tournamentOptions, new TournamentEntrant(white, whiteName),
        new TournamentEntrant(black, blackName));
}

var match = PlayMatch(nnAgent, "GreedyNN", randomAgent, "Random");
Console.WriteLine("Before training:");
Console.WriteLine(match.Summary());

var opponent = new CodingAdventureAgent();
nnAgent.TrainAgainst(opponent, 10, turnTimeLimitMs: 1);

Console.WriteLine("After training:");
match = PlayMatch(nnAgent, "GreedyNN", randomAgent, "Random");
Console.WriteLine(match.Summary());
