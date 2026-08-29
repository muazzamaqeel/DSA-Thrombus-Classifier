using System.Collections.Generic;
using Services.AiService.Responses;

namespace UI.LatencyTest;

public sealed class LatencyExecutionResponse
{
    public string? ExecutionProvider { get; set; }
    public string? TimingDevice { get; set; }
}

public sealed class LatencyClassificationResponse : ClassificationResponse
{
    public double FrontalInferenceMilliseconds { get; set; }
    public double LateralInferenceMilliseconds { get; set; }
    public string? TimingDevice { get; set; }
    public string? ExecutionProvider { get; set; }
}

public sealed class LatencyCase
{
    public string CaseName { get; init; } = "";
    public string FrontalPath { get; init; } = "";
    public string LateralPath { get; init; } = "";
    public string Classification { get; set; } = "-";
    public double? InferenceMilliseconds { get; set; }
    public string ExecutionProvider { get; set; } = "-";
    public string TimingDevice { get; set; } = "-";
    public string Status { get; set; } = "Ready";

    public void Reset()
    {
        Classification = "-";
        InferenceMilliseconds = null;
        ExecutionProvider = "-";
        TimingDevice = "-";
        Status = "Ready";
    }
}

public sealed class ViewLatencyMeasurement
{
    public string CaseName { get; init; } = "";
    public string ModelName { get; init; } = "";
    public double LatencyMilliseconds { get; init; }
    public string ExecutionProvider { get; init; } = "-";
    public string Status => "Complete";
}

public sealed class LatencyCaseRunResult
{
    public bool HasThrombus { get; init; }
    public double InferenceMilliseconds { get; init; }
    public string ExecutionProvider { get; init; } = "-";
    public string TimingDevice { get; init; } = "-";
    public IReadOnlyList<ViewLatencyMeasurement> FrontalMeasurements { get; init; } = [];
    public IReadOnlyList<ViewLatencyMeasurement> LateralMeasurements { get; init; } = [];
}

public sealed class LatencyMetricSummary
{
    public double? Mean { get; init; }
    public double? Min { get; init; }
    public double? Max { get; init; }
}
