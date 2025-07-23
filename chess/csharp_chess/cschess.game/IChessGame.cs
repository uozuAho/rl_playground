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

    void MakeMove(Move move);
}
