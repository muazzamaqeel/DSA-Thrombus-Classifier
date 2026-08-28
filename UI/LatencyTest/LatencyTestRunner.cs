using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using Services.AiService;
using Services.AiService.Interpreter;
using Services.AiService.Responses;

namespace UI.LatencyTest;

public sealed class LatencyTestRunner
{
    private readonly LatencyApiClient _latencyApiClient;

    public LatencyTestRunner(
        LatencyApiClient? latencyApiClient = null)
    {
        _latencyApiClient =
            latencyApiClient ?? new LatencyApiClient();
    }

    public async Task<LatencyCaseRunResult> RunCaseAsync(
        LatencyCase latencyCase,
        IReadOnlyCollection<string> modelNames,
        double classificationThreshold,
        Action<ViewLatencyMeasurement, ViewLatencyMeasurement>?
            onFoldCompleted = null)
    {
        // PREPARE:
        // This deliberately happens before the end-to-end stopwatch.
        await AiServiceCommunication.LoadImages(
            latencyCase.FrontalPath,
            latencyCase.LateralPath);

        // MEASURE END-TO-END CLASSIFICATION PATH:
        var caseStopwatch = Stopwatch.StartNew();

        var responses = new List<ClassificationResponse>();
        var backendInferenceTotalMilliseconds = 0.0;

        var executionProvider = "-";
        var timingDevice = "-";
        var timingMethod = "-";

        // MEASURE EVERY FOLD:
        foreach (var modelName in modelNames)
        {
            var response = await _latencyApiClient.ClassifyAsync(
                modelName,
                modelName,
                latencyCase.FrontalPath,
                latencyCase.LateralPath);

            responses.Add(response);
            backendInferenceTotalMilliseconds +=
                response.InferenceMilliseconds;

            executionProvider =
                NormalizeExecutionProvider(response);
            timingDevice =
                string.IsNullOrWhiteSpace(response.TimingDevice)
                    ? "unknown"
                    : response.TimingDevice;
            timingMethod =
                string.IsNullOrWhiteSpace(response.TimingMethod)
                    ? "unknown"
                    : response.TimingMethod;

            var frontalMeasurement =
                new ViewLatencyMeasurement
                {
                    CaseName = latencyCase.CaseName,
                    ModelName = modelName,
                    LatencyMilliseconds =
                        response.FrontalInferenceMilliseconds,
                    ExecutionProvider = executionProvider,
                    TimingDevice = timingDevice,
                    TimingMethod = timingMethod,
                    Status = "Complete"
                };

            var lateralMeasurement =
                new ViewLatencyMeasurement
                {
                    CaseName = latencyCase.CaseName,
                    ModelName = modelName,
                    LatencyMilliseconds =
                        response.LateralInferenceMilliseconds,
                    ExecutionProvider = executionProvider,
                    TimingDevice = timingDevice,
                    TimingMethod = timingMethod,
                    Status = "Complete"
                };

            onFoldCompleted?.Invoke(
                frontalMeasurement,
                lateralMeasurement);
        }

        // AGGREGATE:
        var averages =
            ResultInterpreter.CalculateCombinedResult(responses);

        var resultInterpreter = new ResultInterpreter
        {
            Threshold = classificationThreshold
        };

        var hasThrombus =
            resultInterpreter.HasThrombus(averages.Item1);

        caseStopwatch.Stop();

        return new LatencyCaseRunResult
        {
            HasThrombus = hasThrombus,
            EndToEndMilliseconds =
                caseStopwatch.Elapsed.TotalMilliseconds,
            BackendInferenceMilliseconds =
                backendInferenceTotalMilliseconds,
            ExecutionProvider = executionProvider,
            TimingDevice = timingDevice,
            TimingMethod = timingMethod
        };
    }

    private static string NormalizeExecutionProvider(
        LatencyClassificationResponse response)
    {
        if (!string.IsNullOrWhiteSpace(
                response.ExecutionProvider))
        {
            return response.ExecutionProvider.ToUpperInvariant();
        }

        if (!string.IsNullOrWhiteSpace(response.TimingDevice) &&
            response.TimingDevice.StartsWith(
                "cuda",
                StringComparison.OrdinalIgnoreCase))
        {
            return "GPU";
        }

        return "CPU";
    }
}
