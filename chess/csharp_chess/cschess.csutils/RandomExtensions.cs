namespace cschess.csutils;

public static class RandomExtensions
{
    public static T Choice<T>(this Random random, IEnumerable<T> source)
    {
        var sourceList = source.ToList();
        var index = random.Next(sourceList.Count);
        return sourceList[index];
    }
}
