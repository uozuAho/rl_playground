using System.Collections.Immutable;
using System.Diagnostics;
using cschess.agents;
using cschess.game;

namespace cschess.tournament;

public record TournamentResults(
    ImmutableList<TournamentEntrant> Entrants,
    ImmutableList<MatchResult> Matches
)
{
    private IEnumerable<TournamentEntrantStats> Stats()
    {
        foreach (var entrant in Entrants)
        {
            var whiteWins = Matches.Where(x => x.White == entrant).Sum(x => x.WhiteWins);
            var blackWins = Matches.Where(x => x.Black == entrant).Sum(x => x.BlackWins);
            var whiteLosses = Matches.Where(x => x.White == entrant).Sum(x => x.BlackWins);
            var blackLosses = Matches.Where(x => x.Black == entrant).Sum(x => x.WhiteWins);
            var whiteDraws = Matches.Where(x => x.White == entrant).Sum(x => x.Draws);
            var blackDraws = Matches.Where(x => x.Black == entrant).Sum(x => x.Draws);

            yield return new TournamentEntrantStats(
                entrant,
                whiteWins,
                whiteLosses,
                whiteDraws,
                blackWins,
                blackLosses,
                blackDraws
            );
        }
    }

    public void PrintStats()
    {
        Console.WriteLine("entrant             score    WLD     WWLD      BWLD");
        foreach (var s in Stats().OrderByDescending(Score))
        {
            Console.WriteLine(
                $"{s.Entrant.Name, -20}{Score(s), 5:0.00}    {s.Wins}/{s.Losses}/{s.Draws}   "
                    + $"{s.WhiteWins}/{s.WhiteLosses}/{s.WhiteDraws}     "
                    + $"{s.BlackWins}/{s.BlackLosses}/{s.BlackDraws}"
            );
        }
    }

    private static double Score(TournamentEntrantStats stats)
    {
        return (stats.Wins - stats.Losses) / (double)stats.Matches;
    }
}

public record TournamentEntrantStats(
    TournamentEntrant Entrant,
    int WhiteWins,
    int WhiteLosses,
    int WhiteDraws,
    int BlackWins,
    int BlackLosses,
    int BlackDraws
)
{
    public int WhiteMatches => WhiteWins + WhiteLosses + WhiteDraws;
    public int BlackMatches => BlackWins + BlackLosses + BlackDraws;
    public int Matches => WhiteMatches + BlackMatches;
    public int Wins => WhiteWins + BlackWins;
    public int Draws => WhiteDraws + BlackDraws;
    public int Losses => WhiteLosses + BlackLosses;
}

public record TournamentOptions(int NumGamesPerMatch, TimeSpan TurnTimeLimit);

public record TournamentEntrant(IChessAgent Agent, string Name);

public record MatchResult(
    TournamentEntrant White,
    TournamentEntrant Black,
    ImmutableList<GameResult> Games
)
{
    public int Draws => Games.Count(x => x.IsDraw);

    public int WhiteWins => Games.Count(x => x.WhiteWon);

    public int BlackWins => Games.Count - WhiteWins - Draws;

    public string Summary()
    {
        var numGames = Games.Count;
        var avgHalfmoves = Games.Sum(x => x.Halfmoves) / numGames;
        var avgGameTime = TimeSpan.FromSeconds(
            Games.Average(x => x.TotalTime.TotalSeconds) / numGames
        );
        var whiteWins = WhiteWins;
        var draws = Draws;
        var blackWins = BlackWins;
        return $"{White.Name} vs {Black.Name}: WLD {whiteWins}/{blackWins}/{draws}. "
            + $"Avg halfmoves: {avgHalfmoves}. Avg game time: {avgGameTime.TotalSeconds:#.###}s.";
    }
}

public record GameResult(
    string FinalState,
    int Halfmoves,
    bool IsDraw,
    bool WhiteWon,
    TimeSpan TotalTime
);

public static class Tournament
{
    public static TournamentResults RunWith(
        TournamentOptions options,
        params TournamentEntrant[] entrants
    )
    {
        var matches = new List<MatchResult>();

        Console.WriteLine(
            $"""
            Running Tournament. Setup:
              - entrants: {entrants.Length}
              - games per match: {options.NumGamesPerMatch}
              - turn time limit (s): {options.TurnTimeLimit.TotalSeconds:#.###}
            """
        );

        for (var i = 0; i < entrants.Length; i++)
        {
            for (var j = 0; j < entrants.Length; j++)
            {
                if (i == j)
                    continue;
                var white = entrants[i];
                var black = entrants[j];

                Console.WriteLine($"Match {i + 1}/{entrants.Length}: {white.Name} vs {black.Name}");

                var matchResult = PlaySingleMatch(options, white, black);
                matches.Add(matchResult);
            }
        }

        return new TournamentResults(entrants.ToImmutableList(), matches.ToImmutableList());
    }

    public static MatchResult PlaySingleMatch(
        TournamentOptions options,
        TournamentEntrant white,
        TournamentEntrant black
    )
    {
        var results = new List<GameResult>(options.NumGamesPerMatch);

        for (var k = 0; k < options.NumGamesPerMatch; k++)
        {
            var result = PlayGame(white.Agent, black.Agent, turnTimeLimit: options.TurnTimeLimit);
            results.Add(result);
        }

        var matchResult = new MatchResult(white, black, results.ToImmutableList());
        return matchResult;
    }

    private static GameResult PlayGame(IChessAgent white, IChessAgent black, TimeSpan turnTimeLimit)
    {
        var game = StuffFactory.CreateGame();

        var stopwatch = Stopwatch.StartNew();
        while (!game.IsGameOver())
        {
            var move =
                game.Turn() == Color.White
                    ? white.NextMove(game, turnTimeLimit)
                    : black.NextMove(game, turnTimeLimit);

            game.MakeMove(move);
        }
        stopwatch.Stop();

        var gameState = game.GameStatus();

        return new GameResult(
            FinalState: gameState.Description,
            Halfmoves: game.HalfmoveCount(),
            IsDraw: gameState is { IsInProgress: false, Winner: null },
            WhiteWon: gameState.Winner == Color.White,
            TotalTime: stopwatch.Elapsed
        );
    }
}

internal class StuffFactory
{
    public static IChessGame CreateGame()
    {
        return CodingAdventureChessGame.StandardGame();
    }
}
