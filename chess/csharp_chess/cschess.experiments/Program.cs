using cschess.agents;
using cschess.experiments;
using cschess.tournament;

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
    var device =
        args.Contains("cpu") ? "cpu"
        : args.Contains("gpu") ? "gpu"
        : "cpu";
    Console.WriteLine($"Using device: {device}");
    var nnAgent = new GreedyNnAgent(device: device);
    var randomAgent = new RandomAgent();

    var opponent = new CodingAdventureAgent();
    nnAgent.TrainAgainst(opponent, 100, turnTimeLimitMs: 1, epCallback: Eval);
    return;

    MatchResult PlayMatch(IChessAgent white, string whiteName, IChessAgent black, string blackName)
    {
        var tournamentOptions = new TournamentOptions(
            NumGamesPerMatch: 5,
            TurnTimeLimit: TimeSpan.FromMilliseconds(10)
        );
        return Tournament.PlaySingleMatch(
            tournamentOptions,
            new TournamentEntrant(white, whiteName),
            new TournamentEntrant(black, blackName)
        );
    }

    void Eval(List<EpisodeStats> episodes)
    {
        // eval against random every 10
        if (episodes.Count % 10 != 0) return;

        var match = PlayMatch(nnAgent, "GreedyNN", randomAgent, "Random");
        Console.WriteLine("Eval vs random:");
        Console.WriteLine(match.Summary());
    }
}
