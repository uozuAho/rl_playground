using Chess.Core;

namespace cschess.game;

public class CodingAdventureChessGame
{
    private readonly Board _board;
    private readonly MoveGenerator _moveGenerator;

    public static CodingAdventureChessGame StandardGame()
    {
        return new CodingAdventureChessGame(Board.CreateBoard());
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
        // todo: perf: method group, linq
        return _moveGenerator.GenerateMoves(_board).ToArray().Select(Move.From);
    }

    public void MakeMove(Move move)
    {
        _board.MakeMove(move.CoreMove);
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
