using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Services.AiService.Interpreter;
using Services.AiService.Responses;

namespace UI.LatencyTest;

public sealed class LatencyTestRunner
{
    private readonly LatencyApiClient _api = new();

    private LatencyExecutionResponse? _execution;

    public Task<LatencyBackendInfo> GetInfoAsync() => _api.GetInfoAsync();

    public async Task<LatencyExecutionResponse> ConfigureExecutionAsync(
        string mode, string modelFolder, int deviceIndex = 0)
    {
        _execution = await _api.ConfigureExecutionAsync(mode, modelFolder, deviceIndex);
        return _execution;
    }

    public async Task<LatencyCaseRunResult> RunCaseAsync(
        LatencyCase latencyCase,
        IReadOnlyCollection<string> modelNames,
        double threshold)
    {
        var caseWatch = Stopwatch.StartNew();
        if (_execution is null)
            throw new InvalidOperationException("Configure the execution device before testing.");
        var preparation = await _api.PrepareImagesAsync(
            latencyCase.FrontalPath, latencyCase.LateralPath);

        try
        {
            var responses = new List<ClassificationResponse>();
            var frontal = new List<ViewLatencyMeasurement>();
            var lateral = new List<ViewLatencyMeasurement>();
            var totalMs = 0.0;
            var execution = "-";
            var device = "-";
            var timingMethod = "";
            var residencies = new HashSet<string>();

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
                timingMethod = response.TimingMethod;
                residencies.Add(response.ModelResidency);

                frontal.Add(new ViewLatencyMeasurement
                {
                    CaseName = latencyCase.CaseName,
                    CaseKey = latencyCase.FrontalPath,
                    ModelName = modelName,
                    ModelOutput = (response.OutputFrontal ??
                        throw new InvalidOperationException("Missing frontal model output.")).Single(),
                    BenchmarkJson = response.FrontalBenchmarkJson,
                    LatencyMilliseconds = response.FrontalInferenceMilliseconds,
                    ExecutionProvider = execution
                });

                lateral.Add(new ViewLatencyMeasurement
                {
                    CaseName = latencyCase.CaseName,
                    CaseKey = latencyCase.FrontalPath,
                    ModelName = modelName,
                    ModelOutput = (response.OutputLateral ??
                        throw new InvalidOperationException("Missing lateral model output.")).Single(),
                    BenchmarkJson = response.LateralBenchmarkJson,
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
                TimingMethod = timingMethod,
                ModelResidency = string.Join("; ", residencies),
                EnvironmentJson = _execution.EnvironmentJson,
                PreparationJson = preparation.PreparationJson,
                ClientCaseMilliseconds = caseWatch.Elapsed.TotalMilliseconds,
                ModelOutput = combined.Item1,
                Threshold = threshold,
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
