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
    public int File => _file;
    public int Rank => _rank;

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

public record Move(Square From, Square To);

public interface IChessGame
{
    bool IsGameOver();
    GameState GameState();
    IEnumerable<Move> LegalMoves();
    PieceType? PieceAt(Square square);
    Color ColorAt(Square square);
    int FullmoveCount();
    int HalfmoveCount();
    Color Turn();

    void MakeMove(Move move);
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
