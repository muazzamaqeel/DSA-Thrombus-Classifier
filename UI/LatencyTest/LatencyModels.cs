using System.Collections.Generic;
using Services.AiService.Responses;

namespace UI.LatencyTest;

public sealed class LatencyBackendInfo
{
    public string? BackendRevision { get; set; }
    public List<LatencyGpuDevice> Devices { get; set; } = [];
}

public sealed class LatencyGpuDevice
{
    public int Index { get; set; }
    public string Name { get; set; } = "";
}

public sealed class LatencyPreparationResponse
{
    public string PreparationJson { get; set; } = "";
}

public sealed class LatencyExecutionResponse
{
    public string? BackendRevision { get; set; }
    public string? RunId { get; set; }
    public string EnvironmentJson { get; set; } = "";
    public string? ExecutionProvider { get; set; }
    public string? TimingDevice { get; set; }
}

public sealed class LatencyClassificationResponse : ClassificationResponse
{
    public string TimingMethod { get; set; } = "";
    public string ModelResidency { get; set; } = "";
    public string FrontalBenchmarkJson { get; set; } = "";
    public string LateralBenchmarkJson { get; set; } = "";
    public double FrontalForwardMilliseconds { get; set; }
    public double LateralForwardMilliseconds { get; set; }
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
    public string TimingMethod { get; set; } = "";
    public string ModelResidency { get; set; } = "";
    public string EnvironmentJson { get; set; } = "";
    public string PreparationJson { get; set; } = "";
    public double? ClientCaseMilliseconds { get; set; }
    // The exact combined score passed to HasThrombus, before thresholding.
    public double? ModelOutput { get; set; }
    public double? Threshold { get; set; }
    public double? ForwardMilliseconds { get; set; }
    public double? InferenceMilliseconds { get; set; }
    public string ExecutionProvider { get; set; } = "-";
    public string TimingDevice { get; set; } = "-";
    public string Status { get; set; } = "Ready";

    public void Reset()
    {
        Classification = "-";
        TimingMethod = ModelResidency = EnvironmentJson = PreparationJson = "";
        ClientCaseMilliseconds = null;
        ModelOutput = null;
        Threshold = null;
        InferenceMilliseconds = null;
        ForwardMilliseconds = null;
        ExecutionProvider = "-";
        TimingDevice = "-";
        Status = "Ready";
    }
}

public sealed class ViewLatencyMeasurement
{
    public string CaseName { get; init; } = "";
    // Full frontal file path: distinct sequences may share the displayed case name.
    public string CaseKey { get; init; } = "";
    public string ModelName { get; init; } = "";
    public string BenchmarkJson { get; init; } = "";
    public double ModelOutput { get; init; }
    public double LatencyMilliseconds { get; init; }
    public double ForwardMilliseconds { get; init; }
    public string ExecutionProvider { get; init; } = "-";
    public string Status => "Complete";
}

public sealed class LatencyCaseRunResult
{
    public string TimingMethod { get; init; } = "";
    public string ModelResidency { get; init; } = "";
    public string EnvironmentJson { get; init; } = "";
    public string PreparationJson { get; init; } = "";
    public double ClientCaseMilliseconds { get; init; }
    public bool HasThrombus { get; init; }
    public double ModelOutput { get; init; }
    public double Threshold { get; init; }
    public double InferenceMilliseconds { get; init; }
    public double ForwardMilliseconds { get; init; }
    public string ExecutionProvider { get; init; } = "-";
    public string TimingDevice { get; init; } = "-";
    public IReadOnlyList<ViewLatencyMeasurement> FrontalMeasurements { get; init; } = [];
    public IReadOnlyList<ViewLatencyMeasurement> LateralMeasurements { get; init; } = [];
}

public sealed class LatencyMetricSummary
{
    public int Count { get; init; }
    public double? StandardDeviation { get; init; }
    public double? Mean { get; init; }
    public double? Min { get; init; }
    public double? Max { get; init; }
}
