using Chess.Core;

namespace cschess.game;

/// <summary>
/// Wraps the coding adventure impl in an easy-to-use wrapper, while trying
/// to keep it fast.
/// </summary>
public class CodingAdventureChessGame : IChessGame
{
    public Board InternalBoard => _board;

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

    public IChessGame Copy()
    {
        return new CodingAdventureChessGame(Board.CreateBoard(_board));
    }

    public bool IsGameOver()
    {
        return Arbiter.GetGameState(_board) != GameResult.InProgress;
    }

    public GameStatus GameStatus()
    {
        var state = Arbiter.GetGameState(_board);
        var winner =
            Arbiter.IsDrawResult(state) ? (Color?)null
            : Arbiter.IsWhiteWinsResult(state) ? Color.White
            : Color.Black;

        return new GameStatus(
            Description: state.ToString(),
            IsInProgress: state == GameResult.InProgress,
            Winner: winner
        );
    }

    public IEnumerable<Move> LegalMoves()
    {
        return _moveGenerator.GenerateMoves(_board).ToArray().Select(ToMyMove).Distinct();
    }

    public void MakeMove(Move move)
    {
        _board.MakeMove(ToCoreMove(move));
    }

    public PieceType? PieceAt(Square square)
    {
        var pieceInt = _board.Square[ToIndex(square)];

        return Piece.PieceType(pieceInt) switch
        {
            Piece.None => null,
            Piece.Pawn => PieceType.Pawn,
            Piece.Knight => PieceType.Knight,
            Piece.Bishop => PieceType.Bishop,
            Piece.Rook => PieceType.Rook,
            Piece.Queen => PieceType.Queen,
            Piece.King => PieceType.King,
            _ => throw new ArgumentOutOfRangeException(nameof(square), square, null),
        };
    }

    public Color ColorAt(Square square)
    {
        if (!PieceAt(square).HasValue)
            return Color.None;

        var pieceInt = _board.Square[ToIndex(square)];

        return Piece.PieceColour(pieceInt) switch
        {
            Piece.White => Color.White,
            Piece.Black => Color.Black,
            _ => throw new ArgumentOutOfRangeException(nameof(square), square, null),
        };
    }

    public int FullmoveCount()
    {
        return HalfmoveCount() / 2;
    }

    public int HalfmoveCount()
    {
        return _board.PlyCount;
    }

    public Color Turn()
    {
        return _board.IsWhiteToMove ? Color.White : Color.Black;
    }

    public string Fen()
    {
        return _board.CurrentFen;
    }

    public void Undo()
    {
        _board.UnmakeMove(_board.AllGameMoves[^1]);
    }

    public static Move ToMyMove(Chess.Core.Move move)
    {
        var from = new Coord(move.StartSquare);
        var to = new Coord(move.TargetSquare);
        var fromSq = Square.Rank0File0(from.rankIndex, from.fileIndex);
        var toSq = Square.Rank0File0(to.rankIndex, to.fileIndex);
        return new Move(fromSq, toSq);
    }

    private static Chess.Core.Move ToCoreMove(Move move)
    {
        var from = new Coord(move.From.File0, move.From.Rank0);
        var to = new Coord(move.To.File0, move.To.Rank0);
        return new Chess.Core.Move(from.SquareIndex, to.SquareIndex);
    }

    private static int ToIndex(Square square)
    {
        return new Coord(square.File0, square.Rank0).SquareIndex;
    }
}
