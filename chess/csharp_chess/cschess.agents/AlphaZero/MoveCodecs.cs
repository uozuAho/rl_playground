using System.Diagnostics;
using cschess.csutils;
using cschess.game;

namespace cschess.agents.AlphaZero;

/// <summary>
/// Convert between chess states, moves / NN outputs etc.
/// </summary>
public interface ICodec
{
    int ActionSize { get; }
    int Move2Int(Move move);
    Dictionary<Move, float> Probdist2Dict(float[] probdist, IChessGame state);
    public float[] Dict2Probdist(Dictionary<Move, float> moveProbs);
    float[,,,] States2Array(IEnumerable<IChessGame> games);
    float[,,] State2Array(IChessGame game);
    float[,] Probs2Array(IEnumerable<Dictionary<Move, float>> moveProbs, ICodec codec);
    float[,] Values2Array(IEnumerable<float> values);
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
        Debug.Assert(Maths.IsProbDist(probdist));
        var moves = state.LegalMoves().ToList();
        Debug.Assert(moves.Count > 0);
        var probs = moves.Select(m => probdist[Move2Int(m)]).ToList();
        var sum = probs.Sum();
        probs = probs.Select(x => x / sum).ToList();
        Debug.Assert(Maths.IsProbDist(probs));
        return moves.Zip(probs).ToDictionary(x => x.First, x => x.Second);
    }

    public float[] Dict2Probdist(Dictionary<Move, float> moveProbs)
    {
        var probs = new float[ActionSize];
        foreach (var (move, prob) in moveProbs)
        {
            probs[Move2Int(move)] = prob;
        }
        return probs;
    }

    public float[,,,] States2Array(IEnumerable<IChessGame> games)
    {
        var gamesList = games.ToList();
        Debug.Assert(gamesList.Count > 0);
        var batch = new float[gamesList.Count, 8, 8, 8];
        for (var b = 0; b < gamesList.Count; b++)
        {
            var arr = State2Array(gamesList[b]);
            for (var i = 0; i < 8; i++)
            for (var j = 0; j < 8; j++)
            for (var k = 0; k < 8; k++)
                batch[b, i, j, k] = arr[i, j, k];
        }
        return batch;
    }

    public float[,,] State2Array(IChessGame game)
    {
        var state = new float[8, 8, 8];

        for (var rank = 0; rank < 8; rank++)
        {
            for (var file = 0; file < 8; file++)
            {
                var square = Square.Rank0File0(rank, file);
                var piece = game.PieceAt(square);
                if (piece == null)
                    continue;

                var sign = game.ColorAt(square) == Color.White ? 1f : -1f;

                var layer = PieceLayer(piece.Value);
                state[layer, rank, file] = sign;
            }
        }

        var moveValue = 1f / (game.FullmoveCount() + 1);
        for (var r = 0; r < 8; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                state[6, r, c] = moveValue;
            }
        }

        var turnValue = game.Turn() == Color.White ? 1f : -1f;
        for (var c = 0; c < 8; c++)
        {
            state[6, 0, c] = turnValue;
        }

        for (var r = 0; r < 8; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                state[7, r, c] = 1f;
            }
        }

        return state;
    }

    public float[,] Probs2Array(IEnumerable<Dictionary<Move, float>> moveProbs, ICodec codec)
    {
        var targetProbs = moveProbs.Select(codec.Dict2Probdist).ToList();
        var nums = new float[targetProbs.Count, codec.ActionSize];
        for (var i = 0; i < targetProbs.Count; i++)
        {
            for (var j = 0; j < codec.ActionSize; j++)
            {
                nums[i, j] = targetProbs[i][j];
            }
        }

        return nums;
    }

    public float[,] Values2Array(IEnumerable<float> values)
    {
        var vList = values.ToList();
        var outf = new float[vList.Count, 1];
        for (var i = 0; i < vList.Count; i++)
        {
            outf[i, 0] = vList[i];
        }

        return outf;
    }

    private static int PieceLayer(PieceType piece)
    {
        return piece switch
        {
            PieceType.Pawn => 0,
            PieceType.Rook => 1,
            PieceType.Knight => 2,
            PieceType.Bishop => 3,
            PieceType.Queen => 4,
            PieceType.King => 5,
            _ => throw new ArgumentOutOfRangeException(nameof(piece), piece, null),
        };
    }

    private static int SquareToInt(Square square)
    {
        return square.Rank0 * 8 + square.File0;
    }

    private static Square IntToSquare(int value)
    {
        return Square.Rank0File0(value >> 3, value & 0b0111);
    }
}
