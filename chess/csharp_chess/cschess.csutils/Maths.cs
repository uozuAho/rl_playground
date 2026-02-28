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

    public static Dictionary<T, float> AddDirichletNoise<T>(
        Dictionary<T, float> simPeval,
        double alpha,
        double epsilon
    )
        where T : notnull
    {
        var nVals = AddDirichletNoise(
            simPeval.Values.Select(x => (double)x).ToArray(),
            alpha,
            epsilon
        );
        return simPeval.Keys.Zip(nVals).ToDictionary(x => x.First, x => (float)x.Second);
    }

    private static double[] AddDirichletNoise(double[] vals, double alpha, double epsilon)
    {
        var noisy = DirichletNoise.AddDirichletNoise(vals, alpha, epsilon);
        Debug.Assert(IsProbDist(noisy));
        return noisy;
    }
}

internal static class DirichletNoise
{
    private static readonly Random Rng = new Random();

    public static double[] AddDirichletNoise(double[] input, double alpha, double epsilon)
    {
        if (input == null || input.Length == 0)
            throw new ArgumentException("Input array must not be null or empty.");

        if (alpha <= 0)
            throw new ArgumentException("Alpha must be > 0.");

        if (epsilon is < 0 or > 1)
            throw new ArgumentException("Epsilon must be in [0,1].");

        var n = input.Length;
        var noise = SampleDirichlet(n, alpha);
        var result = new double[n];

        for (var i = 0; i < n; i++)
        {
            result[i] = (1 - epsilon) * input[i] + epsilon * noise[i];
        }

        Normalize(result);
        return result;
    }

    private static double[] SampleDirichlet(int size, double alpha)
    {
        var samples = new double[size];
        var sum = 0.0;

        for (var i = 0; i < size; i++)
        {
            samples[i] = SampleGamma(alpha, 1.0);
            sum += samples[i];
        }

        for (var i = 0; i < size; i++)
        {
            samples[i] /= sum;
        }

        return samples;
    }

    private static double SampleGamma(double shape, double scale)
    {
        if (shape < 1.0)
        {
            var u = Rng.NextDouble();
            return SampleGamma(shape + 1.0, scale) * Math.Pow(u, 1.0 / shape);
        }

        var d = shape - 1.0 / 3.0;
        var c = 1.0 / Math.Sqrt(9.0 * d);

        while (true)
        {
            var x = SampleStandardNormal();
            var v = 1.0 + c * x;
            if (v <= 0)
                continue;

            v = v * v * v;
            var u = Rng.NextDouble();

            if (u < 1 - 0.0331 * x * x * x * x)
                return scale * d * v;

            if (Math.Log(u) < 0.5 * x * x + d * (1 - v + Math.Log(v)))
                return scale * d * v;
        }
    }

    private static double SampleStandardNormal()
    {
        var u1 = Rng.NextDouble();
        var u2 = Rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static void Normalize(double[] array)
    {
        var sum = 0.0;
        foreach (var v in array)
            sum += v;

        if (sum == 0)
            throw new InvalidOperationException("Cannot normalize zero-sum array.");

        for (var i = 0; i < array.Length; i++)
            array[i] /= sum;
    }
}
