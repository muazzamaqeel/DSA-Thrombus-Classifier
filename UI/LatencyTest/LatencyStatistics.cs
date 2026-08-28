using System.Collections.Generic;
using System.Linq;

namespace UI.LatencyTest;

public static class LatencyStatistics
{
    public static LatencyMetricSummary Calculate(
        IEnumerable<double> values)
    {
        var collected = values.ToList();

        if (collected.Count == 0)
        {
            return new LatencyMetricSummary();
        }

        return new LatencyMetricSummary
        {
            Mean = collected.Average(),
            Min = collected.Min(),
            Max = collected.Max()
        };
    }
}
