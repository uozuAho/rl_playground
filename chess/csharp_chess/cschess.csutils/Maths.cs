using System.Diagnostics;

namespace cschess.csutils;

public static class Maths
{
    public static bool IsProbDist(IEnumerable<double> p)
    {
        var plist = p.ToList();
        return plist.All(x => x is >= 0.0 and <= 1.0) && Math.Abs(plist.Sum() - 1.0) < 0.01;
    }

    public static bool IsProbDist(IEnumerable<float> p)
    {
        var plist = p.ToList();
        return plist.All(x => x is >= 0.0f and <= 1.0f) && Math.Abs(plist.Sum() - 1.0) < 0.01;
    }

    public static Dictionary<T, float> Heat<T>(Dictionary<T, float> dictionary, double temperature)
        where T : notnull
    {
        var keys = dictionary.Keys.ToList();
        var values = Heat(dictionary.Values.ToArray(), temperature);
        return keys.Zip(values).ToDictionary(x => x.First, x => x.Second);
    }

    private static float[] Heat(float[] vals, double temperature)
    {
        Debug.Assert(IsProbDist(vals));
        var newVals = vals.Select(x => (float)Math.Pow(x, 1 / temperature)).ToArray();
        var sum = newVals.Sum();
        newVals = newVals.Select(x => x / sum).ToArray();
        Debug.Assert(IsProbDist(newVals));
        return newVals;
    }
}
