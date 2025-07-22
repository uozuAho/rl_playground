using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.ObjectPool;
using Rudzoft.ChessLib;
using Rudzoft.ChessLib.Enums;
using Rudzoft.ChessLib.Hash;
using Rudzoft.ChessLib.MoveGeneration;
using Rudzoft.ChessLib.Types;
using Rudzoft.ChessLib.Validation;

namespace cschess.game;

/// <summary>
/// WIP chess game wrapper, using https://github.com/rudzen/ChessLib
/// </summary>
public class ChessLibChessGame
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ObjectPool<MoveList> _moveListPool;
    private readonly IPosition _pos;

    public static ChessLibChessGame StandardGame()
    {
        return new ChessLibChessGame();
    }

    private ChessLibChessGame()
    {
        // todo: move to static?
        _serviceProvider = new ServiceCollection()
            .AddTransient<IBoard, Board>()
            .AddSingleton<IValues, Values>()
            .AddSingleton<IRKiss, RKiss>()
            .AddSingleton<IZobrist, Zobrist>()
            .AddSingleton<ICuckoo, Cuckoo>()
            .AddSingleton<IPositionValidator, PositionValidator>()
            .AddTransient<IPosition, Position>()
            .AddSingleton<ObjectPoolProvider, DefaultObjectPoolProvider>()
            .AddSingleton(static serviceProvider =>
            {
                var provider = serviceProvider.GetRequiredService<ObjectPoolProvider>();
                var policy = new DefaultPooledObjectPolicy<MoveList>();
                return provider.Create(policy);
            })
            .BuildServiceProvider();

        _moveListPool = _serviceProvider.GetRequiredService<ObjectPool<MoveList>>();

        _pos = _serviceProvider.GetRequiredService<IPosition>();
    }

    // public bool IsGameOver()
    // {
    //     return GameEndType(pos) != GameEndTypes.None;
    // }

    public IEnumerable<T> LegalMoves<T>()
    {
        throw new NotImplementedException();
    }

    public void MakeMove(object move)
    {
        throw new NotImplementedException();
    }

    private bool IsGameOver(IPosition pos)
    {
        return GameEndType(pos) != GameEndTypes.None;
    }

    private GameEndTypes GameEndType(IPosition pos)
    {
        var gameEndType = GameEndTypes.None;
        if (pos.IsRepetition)
            gameEndType |= GameEndTypes.Repetition;
        if (pos.Rule50 >= 100)
            gameEndType |= GameEndTypes.FiftyMove;

        // todo: not sure if this is correct usage
        var moveList = _moveListPool.Get();
        moveList.Generate(in pos);

        var moves = moveList.Get();

        if (moves.IsEmpty)
            gameEndType |= GameEndTypes.Pat;

        _moveListPool.Return(moveList);

        return gameEndType;
    }
}
