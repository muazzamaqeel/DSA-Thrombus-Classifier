using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace UI.LatencyTest;

public sealed class LatencyApiClient
{
    private static readonly HttpClient Client = new()
    {
        BaseAddress = new Uri($"http://{Services.Configuration.AiServiceUrl}/"),
        Timeout = TimeSpan.FromMinutes(5)
    };

    private string? _runId;
    public const string RequiredBackendRevision = "latency-v6-isolated-single-pass";

    public async Task<LatencyBackendInfo> GetInfoAsync()
    {
        using var response = await Client.GetAsync("/AiService/LatencyInfo");
        response.EnsureSuccessStatusCode();
        var info = JsonConvert.DeserializeObject<LatencyBackendInfo>(await response.Content.ReadAsStringAsync())
            ?? throw new InvalidOperationException("Invalid backend information response.");
        if (info.BackendRevision != RequiredBackendRevision)
            throw new InvalidOperationException("Update and restart the configured Python backend with the v6 files.");
        return info;
    }

    public async Task<LatencyExecutionResponse> ConfigureExecutionAsync(
        string mode, string modelFolder, int deviceIndex = 0)
    {
        _runId = Guid.NewGuid().ToString("N");
        using var response = await PostAsync(
            "/AiService/LatencyExecutionMode",
            new { Mode = mode, ModelFolder = modelFolder, DeviceIndex = deviceIndex, RunId = _runId });

        var json = await response.Content.ReadAsStringAsync();
        var execution = JsonConvert.DeserializeObject<LatencyExecutionResponse>(json)
               ?? throw new InvalidOperationException(
                   "Invalid latency execution-mode response.");
        if (execution.BackendRevision != RequiredBackendRevision || string.IsNullOrWhiteSpace(execution.RunId))
            throw new InvalidOperationException("Update and restart the configured Python backend with the v6 files.");
        _runId = execution.RunId;
        return execution;
    }

    public async Task<LatencyPreparationResponse> PrepareImagesAsync(string frontal, string lateral)
    {
        using var response = await PostAsync("/AiService/LatencyPrepareImages", new
        {
            PathFrontal = frontal, PathLateral = lateral, RunId = _runId
        });
        return JsonConvert.DeserializeObject<LatencyPreparationResponse>(await response.Content.ReadAsStringAsync())
            ?? throw new InvalidOperationException("Invalid preparation response.");
    }

    public Task ReleasePreparedImagesAsync(string frontal, string lateral) =>
        PostNoContentAsync("/AiService/LatencyReleaseImages", new
        {
            PathFrontal = frontal,
            PathLateral = lateral,
            RunId = _runId
        });

    public async Task<LatencyClassificationResponse> ClassifyAsync(
        string modelName, string frontal, string lateral)
    {
        using var response = await PostAsync(
            "/AiService/LatencyClassification",
            new
            {
                ModelName = modelName,
                PathFrontal = frontal,
                PathLateral = lateral,
                RunId = _runId
            });

        var json = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<LatencyClassificationResponse>(json)
               ?? throw new InvalidOperationException(
                   "Invalid latency classification response.");
    }

    public Task CancelAsync() => _runId is null ? Task.CompletedTask :
        PostNoContentAsync("/AiService/LatencyCancel", new { RunId = _runId });

    public async Task EndAsync()
    {
        if (_runId is null) return;
        await PostNoContentAsync("/AiService/LatencyEnd", new { RunId = _runId });
        _runId = null;
    }

    private static async Task PostNoContentAsync(string uri, object payload)
    {
        using var response = await PostAsync(uri, payload);
    }

    private static async Task<HttpResponseMessage> PostAsync(string uri, object payload)
    {
        using var data = new StringContent(
            JsonConvert.SerializeObject(payload), Encoding.UTF8, "application/json");
        var response = await Client.PostAsync(uri, data);
        if (!response.IsSuccessStatusCode)
        {
            var message = await response.Content.ReadAsStringAsync();
            var status = response.StatusCode;
            response.Dispose();
            throw new HttpRequestException($"Backend returned {status}: {message}");
        }
        return response;
    }
}
