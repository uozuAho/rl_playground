using System.Collections.Immutable;
using cschess.agents;
using cschess.game;

namespace cschess.tournament;

public record TournamentResults(ImmutableList<MatchResult> Matches);

public record TournamentOptions(
    int NumGamesPerMatch,
    TimeSpan TurnTimeLimit);

public record TournamentEntrant(IChessAgent Agent, string Name);

public record MatchResult(
    TournamentEntrant White,
    TournamentEntrant Black,
    ImmutableList<GameResult> Games
);

public record GameResult(string FinalState, int Halfmoves, bool IsDraw, bool WhiteWon);

public class Tournament
{
    public static TournamentResults RunWith(
        TournamentOptions options,
        params TournamentEntrant[] entrants
    )
    {
        var matches = new List<MatchResult>();

        Console.WriteLine(
            $"Running Tournament. {options.NumGamesPerMatch} games per match, {entrants.Length} entrants"
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

                var results = new List<GameResult>(options.NumGamesPerMatch);

                for (var k = 0; k < options.NumGamesPerMatch; k++)
                {
                    var result = PlayGame(white.Agent, black.Agent, turnTimeLimit: options.TurnTimeLimit);
                    results.Add(result);
                }

                var matchResult = new MatchResult(white, black, results.ToImmutableList());
                matches.Add(matchResult);
            }
        }

        return new TournamentResults(matches.ToImmutableList());
    }

    private static GameResult PlayGame(IChessAgent white, IChessAgent black, TimeSpan turnTimeLimit)
    {
        var game = StuffFactory.CreateGame();

        while (!game.IsGameOver())
        {
            var move = game.Turn() == Color.White
                ? white.NextMove(game, turnTimeLimit)
                : black.NextMove(game, turnTimeLimit);

            game.MakeMove(move);
        }

        var gameState = game.GameState();

        return new GameResult(
            FinalState: gameState.Description,
            Halfmoves: game.HalfmoveCount(),
            IsDraw: gameState.IsDraw,
            WhiteWon: gameState.IsWhiteWin
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
