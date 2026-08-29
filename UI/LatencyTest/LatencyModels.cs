using Services.AiService.Responses;

namespace UI.LatencyTest;

public sealed class LatencyClassificationResponse : ClassificationResponse
{
    public double FrontalInferenceMilliseconds { get; set; }
    public double LateralInferenceMilliseconds { get; set; }
    public double InferenceMilliseconds { get; set; }
    public string? TimingDevice { get; set; }
    public string? ExecutionProvider { get; set; }
    public string? TimingMethod { get; set; }
}

public sealed class LatencyCase
{
    public string CaseName { get; set; } = "";
    public string FrontalPath { get; set; } = "";
    public string LateralPath { get; set; } = "";
    public string Classification { get; set; } = "";
    public string LatencyMilliseconds { get; set; } = "";
    public double? LatencyValueMilliseconds { get; set; }
    public string BackendInferenceMilliseconds { get; set; } = "";
    public double? BackendInferenceValueMilliseconds { get; set; }
    public string ExecutionProvider { get; set; } = "";
    public string TimingDevice { get; set; } = "";
    public string TimingMethod { get; set; } = "";
    public string Status { get; set; } = "";

    public void ResetForRun()
    {
        Classification = "-";
        LatencyMilliseconds = "-";
        LatencyValueMilliseconds = null;
        BackendInferenceMilliseconds = "-";
        BackendInferenceValueMilliseconds = null;
        ExecutionProvider = "-";
        TimingDevice = "-";
        TimingMethod = "-";
        Status = "Ready";
    }
}

public sealed class ViewLatencyMeasurement
{
    public string CaseName { get; set; } = "";
    public string ModelName { get; set; } = "";
    public double LatencyMilliseconds { get; set; }
    public string ExecutionProvider { get; set; } = "";
    public string TimingDevice { get; set; } = "";
    public string TimingMethod { get; set; } = "";
    public string Status { get; set; } = "";
}

public sealed class LatencyCaseRunResult
{
    public bool HasThrombus { get; init; }
    public double EndToEndMilliseconds { get; init; }
    public double BackendInferenceMilliseconds { get; init; }
    public string ExecutionProvider { get; init; } = "-";
    public string TimingDevice { get; init; } = "-";
    public string TimingMethod { get; init; } = "-";
}

public sealed class LatencyMetricSummary
{
    public double? Mean { get; init; }
    public double? Min { get; init; }
    public double? Max { get; init; }
}
