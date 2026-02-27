using cschess.game;

namespace cschess.agents.AlphaZero;

public interface ICodec
{
    int ActionSize { get; }
    int Move2Int(Move move);
    Dictionary<Move, float> Probdist2Dict(float[] probdist, IChessGame state);
}

/// <summary>
/// Simple from-to square encoding. No en-passant, promotions etc.
/// </summary>
public class Codec4096 : ICodec
{
    public int ActionSize => 4096;

    public int Move2Int(Move move)
    {
        var from = SquareToInt(move.From);
        var to = SquareToInt(move.To);

        return from * 64 + to;
    }

    public Dictionary<Move, float> Probdist2Dict(float[] probdist, IChessGame state)
    {
        return state.LegalMoves().ToDictionary(x => x, x => probdist[Move2Int(x)]);
    }

    private static int SquareToInt(Square square)
    {
        return square.Rank * 8 + square.File;
    }

    private static Square IntToSquare(int value)
    {
        return Square.FromRankAndFile(value >> 3, value & 0b0111);
    }
}
