using System;
using System.Collections.Generic;

namespace UI.LatencyTest;

public static class LatencyStatistics
{
    public static LatencyMetricSummary Calculate(IEnumerable<double> values)
    {
        var count = 0;
        double mean = 0, m2 = 0;
        var min = double.PositiveInfinity;
        var max = double.NegativeInfinity;
        foreach (var value in values)
        {
            if (double.IsNaN(value) || double.IsInfinity(value) || value < 0)
                throw new ArgumentException("Latency must be finite and non-negative.");
            count++;
            var delta = value - mean;
            mean += delta / count;
            m2 += delta * (value - mean);
            min = Math.Min(min, value);
            max = Math.Max(max, value);
        }
        return new LatencyMetricSummary
        {
            Count = count,
            Mean = count > 0 ? mean : null,
            Min = count > 0 ? min : null,
            Max = count > 0 ? max : null,
            StandardDeviation = count > 1 ? Math.Sqrt(Math.Max(0, m2 / (count - 1))) : null
        };
    }
}
