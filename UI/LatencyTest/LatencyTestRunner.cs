using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Threading;
using Services.AiService.Interpreter;
using Services.AiService.Responses;

namespace UI.LatencyTest;

public sealed class LatencyTestRunner
{
    private readonly LatencyApiClient _api = new();

    private LatencyExecutionResponse? _execution;

    public Task CancelAsync() => _api.CancelAsync();
    public Task EndAsync() => _api.EndAsync();

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
        double threshold,
        CancellationToken cancellationToken = default)
    {
        var caseWatch = Stopwatch.StartNew();
        if (_execution is null)
            throw new InvalidOperationException("Configure the execution device before testing.");
        cancellationToken.ThrowIfCancellationRequested();
        try
        {
            var preparation = await _api.PrepareImagesAsync(latencyCase.FrontalPath, latencyCase.LateralPath);
            cancellationToken.ThrowIfCancellationRequested();
            var responses = new List<ClassificationResponse>();
            var frontal = new List<ViewLatencyMeasurement>();
            var lateral = new List<ViewLatencyMeasurement>();
            var totalMs = 0.0;
            var forwardMs = 0.0;
            var execution = "-";
            var device = "-";
            var timingMethod = "";
            var residencies = new HashSet<string>();

            // Sequential by design: parallel GPU folds would distort latency.
            foreach (var modelName in modelNames)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var response = await _api.ClassifyAsync(
                    modelName,
                    latencyCase.FrontalPath, latencyCase.LateralPath);

                cancellationToken.ThrowIfCancellationRequested();
                responses.Add(response);
                forwardMs += response.FrontalForwardMilliseconds + response.LateralForwardMilliseconds;
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
                    ForwardMilliseconds = response.FrontalForwardMilliseconds,
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
                    ForwardMilliseconds = response.LateralForwardMilliseconds,
                    ExecutionProvider = execution
                });
            }

            var aggregationWatch = Stopwatch.StartNew();
            var combined = ResultInterpreter.CalculateCombinedResult(responses);
            var hasThrombus = new ResultInterpreter { Threshold = threshold }
                .HasThrombus(combined.Item1);
            aggregationWatch.Stop();
            totalMs += aggregationWatch.Elapsed.TotalMilliseconds;

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
                ForwardMilliseconds = forwardMs,
                ExecutionProvider = execution,
                TimingDevice = device,
                FrontalMeasurements = frontal,
                LateralMeasurements = lateral
            };
        }
        catch
        {
            try { await _api.CancelAsync(); } catch (Exception error) { Debug.WriteLine(error); }
            throw;
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
