using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using Services.AiService.Interpreter;
using Services.AiService.Responses;

namespace UI.LatencyTest;

public sealed class LatencyTestRunner
{
    private readonly LatencyApiClient _api = new();

    public Task<LatencyExecutionResponse> ConfigureExecutionAsync(
        string mode, string modelFolder) =>
        _api.ConfigureExecutionAsync(mode, modelFolder);

    public async Task<LatencyCaseRunResult> RunCaseAsync(
        LatencyCase latencyCase,
        IReadOnlyCollection<string> modelNames,
        double threshold)
    {
        await _api.PrepareImagesAsync(
            latencyCase.FrontalPath, latencyCase.LateralPath);

        try
        {
            var responses = new List<ClassificationResponse>();
            var frontal = new List<ViewLatencyMeasurement>();
            var lateral = new List<ViewLatencyMeasurement>();
            var totalMs = 0.0;
            var execution = "-";
            var device = "-";

            // Sequential by design: parallel GPU folds would distort latency.
            foreach (var modelName in modelNames)
            {
                var response = await _api.ClassifyAsync(
                    modelName,
                    latencyCase.FrontalPath, latencyCase.LateralPath);

                responses.Add(response);
                totalMs += response.FrontalInferenceMilliseconds +
                           response.LateralInferenceMilliseconds;
                execution = response.ExecutionProvider?.ToUpperInvariant() ?? "CPU";
                device = response.TimingDevice ?? "unknown";

                frontal.Add(new ViewLatencyMeasurement
                {
                    CaseName = latencyCase.CaseName,
                    ModelName = modelName,
                    LatencyMilliseconds = response.FrontalInferenceMilliseconds,
                    ExecutionProvider = execution
                });

                lateral.Add(new ViewLatencyMeasurement
                {
                    CaseName = latencyCase.CaseName,
                    ModelName = modelName,
                    LatencyMilliseconds = response.LateralInferenceMilliseconds,
                    ExecutionProvider = execution
                });
            }

            var combined = ResultInterpreter.CalculateCombinedResult(responses);
            var hasThrombus = new ResultInterpreter { Threshold = threshold }
                .HasThrombus(combined.Item1);

            return new LatencyCaseRunResult
            {
                HasThrombus = hasThrombus,
                InferenceMilliseconds = totalMs,
                ExecutionProvider = execution,
                TimingDevice = device,
                FrontalMeasurements = frontal,
                LateralMeasurements = lateral
            };
        }
        finally
        {
            try
            {
                await _api.ReleasePreparedImagesAsync(
                    latencyCase.FrontalPath, latencyCase.LateralPath);
            }
            catch (Exception error)
            {
                Debug.WriteLine($"Latency cache cleanup failed: {error.Message}");
            }
        }
    }
}
