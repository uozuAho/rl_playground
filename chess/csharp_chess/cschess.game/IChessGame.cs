namespace cschess.game;

public record GameState(
    string Description,
    bool IsInProgress,
    bool IsDraw,
    bool IsWhiteWin,
    bool IsBlackWin
);

//todo: should this be a struct/record/etc
public class Square
{
    private byte _rank; // row 1-8
    private byte _file; // col a-h

    private Square(int rank, int file)
    {
        _rank = (byte)rank;
        _file = (byte)file;
    }

    public static Square FromRankAndFile(int rank, int file)
    {
        return new Square(rank, file);
    }
}

public record MyMove(Square From, Square To);

public interface IChessGame
{
    bool IsGameOver();
    GameState GameState();
    IEnumerable<MyMove> LegalMoves();
    PieceType? PieceAt(int square);
    Color ColorAt(int square);
    int FullmoveCount();
    int HalfmoveCount();
    Color Turn();

    void MakeMove(MyMove move);
    void Undo();
}

public enum PieceType
{
    Pawn = 1,
    Rook = 2,
    Bishop = 3,
    Knight = 4,
    Queen = 5,
    King = 6,
};

public enum Color
{
    Black = -1,
    None = 0,
    White = 1,
}
