using System.Diagnostics;

namespace cschess.game;

public record GameStatus(string Description, bool IsInProgress, Color? Winner);

[DebuggerDisplay("{ToUci()}")]
public readonly record struct Square
{
    private const string fileChars = "abcdefgh";
    private readonly byte _rank0; // row 1-8, 0-indexed
    private readonly byte _file0; // col a-h, 0-indexed
    public int File0 => _file0;
    public int Rank0 => _rank0;

    private Square(int rank0, int file0)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(rank0, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(rank0, 7);
        ArgumentOutOfRangeException.ThrowIfLessThan(file0, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(file0, 7);
        _rank0 = (byte)rank0;
        _file0 = (byte)file0;
    }

    public static Square Rank0File0(int rank, int file)
    {
        return new Square(rank, file);
    }

    public string ToUci()
    {
        var col = fileChars[_file0];
        var rank = (_rank0 + 1).ToString();
        return col + rank;
    }

    public static Square FromUci(string uci)
    {
        var file0 = fileChars.IndexOf(uci[0]);
        var rank1 = int.Parse(uci.Substring(1, 1));
        return new Square(rank1 - 1, file0);
    }
}

[DebuggerDisplay("{ToUci()}")]
public readonly record struct Move(Square From, Square To)
{
    public static Move FromUci(string uci)
    {
        var fromUci = uci.Substring(0, 2);
        var toUci = uci.Substring(2, 2);
        return new Move(Square.FromUci(fromUci), Square.FromUci(toUci));
    }

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
    string Fen();
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
