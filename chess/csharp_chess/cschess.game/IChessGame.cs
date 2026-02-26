namespace cschess.game;

public record GameStatus(string Description, bool IsInProgress, Color? Winner);

//todo: should this be a struct/record/etc
//todo: make illegal construction impossible
public readonly record struct Square
{
    private const string fileChars = "abcdefgh";
    private readonly byte _rank; // row 1-8, 0-indexed
    private readonly byte _file; // col a-h
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

    public string ToUci()
    {
        var col = fileChars[_file];
        var rank = (_rank + 1).ToString();
        return col + rank;
    }
}

public readonly record struct Move(Square From, Square To)
{
    public string ToUci()
    {
        return From.ToUci() + To.ToUci();
    }
}

public interface IChessGame
{
    bool IsGameOver();
    GameStatus GameStatus();
    IEnumerable<Move> LegalMoves();
    PieceType? PieceAt(Square square);
    Color ColorAt(Square square);
    int FullmoveCount();
    int HalfmoveCount();
    Color Turn();

    void MakeMove(Move move);
    void Undo();
    IChessGame Copy();
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
