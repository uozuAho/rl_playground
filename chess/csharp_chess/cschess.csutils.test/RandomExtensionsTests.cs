using Shouldly;

namespace cschess.csutils.test;

public class RandomExtensionsTests
{
    [Fact]
    public void Weighted_choice_picks_expected()
    {
        var rng = new Random();
        var options = new[] { 0, 1, 2 };
        var weights = new[] { 0.5, 0, 0.5 };
        var counts = new[] { 0, 0, 0 };
        for (var i = 0; i < 100; i++)
        {
            var val = rng.Choice(options, weights);
            counts[val]++;
        }

        counts[0].ShouldBeGreaterThan(0);
        counts[1].ShouldBe(0);
        counts[2].ShouldBeGreaterThan(0);
        (counts[0] + counts[2]).ShouldBe(100);
    }
}
