using System.Diagnostics;

namespace cschess.agents.AlphaZero;

internal class TaskMetrics
{
    private readonly string _name;
    private readonly Stopwatch _sw;
    private TimeSpan _workStarted;
    private TimeSpan _workTime;
    private int _games;
    private int _states;

    private TaskMetrics(string name, Stopwatch sw)
    {
        _name = name;
        _sw = sw;
    }

    public static TaskMetrics StartNew(string name)
    {
        return new TaskMetrics(name, Stopwatch.StartNew());
    }

    public void StartWork() => _workStarted = _sw.Elapsed;

    public void StopWork() => _workTime += _sw.Elapsed - _workStarted;

    public void IncGame() => _games += 1;

    public void IncState() => _states += 1;

    public void IncState(int nStates) => _states += nStates;

    public void PrintSummary()
    {
        var totalTime = _sw.Elapsed;
        var gamesPerSec = _games / totalTime.TotalSeconds;
        var statesPerSec = _states / totalTime.TotalSeconds;
        var util = _workTime / totalTime;
        Console.WriteLine($"{_name}: {_games} games, {_states} states in {totalTime}");
        Console.WriteLine($"{_name}: {gamesPerSec:F2} games/sec, {statesPerSec:F2} states/sec");
        Console.WriteLine($"{_name}: utilisation: {util:F2}");
    }
}
