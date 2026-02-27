using System.Diagnostics;

namespace cschess.csutils;

public static class RandomExtensions
{
    public static T Choice<T>(this Random random, IEnumerable<T> source)
    {
        var sourceList = source.ToList();
        var index = random.Next(sourceList.Count);
        return sourceList[index];
    }

    /// <summary>
    /// Choose item based on probability distribution
    /// </summary>
    /// <param name="random"></param>
    /// <param name="source"></param>
    /// <param name="weights">Assumed to be a probability distribution: [0.0-1.0], sums to 1</param>
    /// <typeparam name="T"></typeparam>
    public static T Choice<T>(
        this Random random,
        IEnumerable<T> source,
        IEnumerable<double> weights
    )
    {
        var total = 0.0;
        using var sourceEnum = source.GetEnumerator();
        var rval = random.NextDouble();
        foreach (var weight in weights)
        {
            total += weight;
            Debug.Assert(weight is >= 0.0 and <= 1.0);
            Debug.Assert(total <= 1.0);
            if (rval < total)
            {
                return sourceEnum.Current;
            }

            sourceEnum.MoveNext();
        }

        Debug.Assert(total >= 0.999);
        throw new Exception("Shouldn't get here. Probably something wrong with args.");
    }

    public static T Choice<T>(this Random random, IEnumerable<T> source, IEnumerable<float> weights)
    {
        return random.Choice(source, weights.Select(x => (double)x));
    }
}
