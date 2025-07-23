using Chess.Core;

namespace cschess.game;

public record GameState(
    string Description,
    bool IsInProgress,
    bool IsDraw,
    bool IsWhiteWin,
    bool IsBlackWin);

public interface IChessGame
{
    bool IsGameOver();
    GameState GameState();
    IEnumerable<Move> LegalMoves();
    PieceType? PieceAt(int square);
    Color ColorAt(int square);
    int FullmoveCount();
    int HalfmoveCount();
    Color Turn();

    /// <summary>
    /// This is just here to make initial implementation easy.
    /// Don't use widely.
    /// </summary>
    Board InternalBoard { get; }

    void MakeMove(Move move);
}
