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
    float[,,,] StatesToNumbers(IEnumerable<IChessGame> games);
    float[,,] StateToNumbers(IChessGame game);
    float[,] ProbsToNumbers(IEnumerable<Dictionary<Move, float>> moveProbs, ICodec codec);
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

    public float[] Dict2Probdist(Dictionary<Move, float> moveProbs)
    {
        var probs = new float[ActionSize];
        foreach (var (move, prob) in moveProbs)
        {
            probs[Move2Int(move)] = prob;
        }
        return probs;
    }

    public float[,,,] StatesToNumbers(IEnumerable<IChessGame> games)
    {
        var gamesList = games.ToList();
        var batch = new float[gamesList.Count, 8, 8, 8];
        for (var b = 0; b < gamesList.Count; b++)
        {
            var arr = StateToNumbers(gamesList[b]);
            for (var i = 0; i < 8; i++)
            for (var j = 0; j < 8; j++)
            for (var k = 0; k < 8; k++)
                batch[b, i, j, k] = arr[i, j, k];
        }
        return batch;
    }

    public float[,,] StateToNumbers(IChessGame game)
    {
        var state = new float[8, 8, 8];

        for (var rank = 0; rank < 8; rank++)
        {
            for (var file = 0; file < 8; file++)
            {
                var square = Square.FromRankAndFile(rank, file);
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

    public float[,] ProbsToNumbers(IEnumerable<Dictionary<Move, float>> moveProbs, ICodec codec)
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
        return square.Rank * 8 + square.File;
    }

    private static Square IntToSquare(int value)
    {
        return Square.FromRankAndFile(value >> 3, value & 0b0111);
    }
}
