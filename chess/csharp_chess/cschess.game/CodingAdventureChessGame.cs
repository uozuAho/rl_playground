using Chess.Core;

namespace cschess.game;

/// <summary>
/// Wraps the coding adventure impl in an easy-to-use wrapper, while trying
/// to keep it fast.
/// </summary>
public class CodingAdventureChessGame
{
    private readonly Board _board;
    private readonly MoveGenerator _moveGenerator;

    public static CodingAdventureChessGame StandardGame()
    {
        return new CodingAdventureChessGame(Board.CreateBoard());
    }

    public static CodingAdventureChessGame FromFen(string fen)
    {
        return new CodingAdventureChessGame(Board.CreateBoard(fen));
    }

    private CodingAdventureChessGame(Board board)
    {
        _board = board;
        _moveGenerator = new MoveGenerator();
    }

    public bool IsGameOver()
    {
        return Arbiter.GetGameState(_board) != GameResult.InProgress;
    }

    public IEnumerable<Move> LegalMoves()
    {
        return _moveGenerator.GenerateMoves(_board).ToArray().Select(Move.From);
    }

    public void MakeMove(Move move)
    {
        _board.MakeMove(move.CoreMove);
    }

    public PieceType? PieceAt(int square)
    {
        var pieceInt = _board.Square[square];

        return Piece.PieceType(pieceInt) switch
        {
            Piece.None => null,
            Piece.Pawn => PieceType.Pawn,
            Piece.Knight => PieceType.Knight,
            Piece.Bishop => PieceType.Bishop,
            Piece.Rook => PieceType.Rook,
            Piece.Queen => PieceType.Queen,
            Piece.King => PieceType.King,
            _ => throw new ArgumentOutOfRangeException(nameof(square), square, null)
        };
    }

    public Color ColorAt(int square)
    {
        if (!PieceAt(square).HasValue) return Color.None;

        var pieceInt = _board.Square[square];

        return Piece.PieceColour(pieceInt) switch
        {
            Piece.White => Color.White,
            Piece.Black => Color.Black,
            _ => throw new ArgumentOutOfRangeException(nameof(square), square, null)
        };
    }

    public int FullmoveCount()
    {
        return _board.PlyCount / 2;
    }

    public Color Turn()
    {
        return _board.IsWhiteToMove ? Color.White : Color.Black;
    }
}

public class Move
{
    internal readonly Chess.Core.Move CoreMove;

    internal static Move From(Chess.Core.Move move)
    {
        return new Move(move);
    }

    private Move(Chess.Core.Move coreMove)
    {
        CoreMove = coreMove;
    }
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
