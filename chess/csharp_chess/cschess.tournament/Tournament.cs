using System.Collections.Immutable;
using System.Diagnostics;
using cschess.agents;
using cschess.game;

namespace cschess.tournament;

public record TournamentResults(ImmutableList<MatchResult> Matches);

public record TournamentOptions(int NumGamesPerMatch, TimeSpan TurnTimeLimit);

public record TournamentEntrant(IChessAgent Agent, string Name);

public record MatchResult(
    TournamentEntrant White,
    TournamentEntrant Black,
    ImmutableList<GameResult> Games
)
{
    public string Summary()
    {
        var numGames = Games.Count;
        var avgHalfmoves = Games.Sum(x => x.Halfmoves) / numGames;
        var avgGameTime = TimeSpan.FromSeconds(
            Games.Average(x => x.TotalTime.TotalSeconds) / numGames
        );
        var whiteWins = Games.Count(x => x.WhiteWon);
        var draws = Games.Count(x => x.IsDraw);
        var blackWins = numGames - whiteWins - draws;
        return $"{White.Name} (white) vs {Black.Name} (black): W/D/L: {whiteWins}/{draws}/{blackWins}. "
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

public class Tournament
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

        // todo: swiss-style rather than round robin
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

        return new TournamentResults(matches.ToImmutableList());
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

        var gameState = game.GameState();

        return new GameResult(
            FinalState: gameState.Description,
            Halfmoves: game.HalfmoveCount(),
            IsDraw: gameState.IsDraw,
            WhiteWon: gameState.IsWhiteWin,
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
