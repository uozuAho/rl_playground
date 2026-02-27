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
}
