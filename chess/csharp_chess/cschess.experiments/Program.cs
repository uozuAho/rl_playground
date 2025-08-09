using cschess.agents;
using cschess.tournament;
using cschess.experiments;

// TrainValueNet();
TrainGreedyNn();

void TrainValueNet()
{
    Console.WriteLine(string.Join(",", args));
    var fenScoresPath = args[0];
    ValueNetworkTrainer.TrainAndTestValueNetwork(fenScoresPath);
}

void TrainGreedyNn()
{
    var device = args.Contains("cpu") ? "cpu" : args.Contains("gpu") ? "gpu" : "cpu";
    Console.WriteLine($"Using device: {device}");
    var nnAgent = new GreedyNnAgent(device: device);
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
}
