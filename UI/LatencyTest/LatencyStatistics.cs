using System;
using System.Collections.Generic;

namespace UI.LatencyTest;

public static class LatencyStatistics
{
    public static LatencyMetricSummary Calculate(IEnumerable<double> values)
    {
        var count = 0;
        var sum = 0.0;
        var min = double.PositiveInfinity;
        var max = double.NegativeInfinity;

        foreach (var value in values)
        {
            count++;
            sum += value;
            min = Math.Min(min, value);
            max = Math.Max(max, value);
        }

        return count == 0
            ? new LatencyMetricSummary()
            : new LatencyMetricSummary
            {
                Mean = sum / count,
                Min = min,
                Max = max
            };
    }
}
