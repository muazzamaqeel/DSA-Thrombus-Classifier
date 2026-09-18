
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
    public const string RequiredBackendRevision = "latency-v7-gpu-batched-single-pass";

    public async Task<LatencyBackendInfo> GetInfoAsync()
    {
        using var response = await Client.GetAsync("/AiService/LatencyInfo");
        await ThrowBackendErrorAsync(response, "/AiService/LatencyInfo");
        var info = JsonConvert.DeserializeObject<LatencyBackendInfo>(await response.Content.ReadAsStringAsync())
            ?? throw new InvalidOperationException("Invalid backend information response.");
        if (info.BackendRevision != RequiredBackendRevision)
            throw new InvalidOperationException(
                $"Backend/UI version mismatch. Backend reports '{info.BackendRevision ?? "<missing>"}', " +
                $"but this UI requires '{RequiredBackendRevision}'. Restart the backend from this project.");
        return info;
    }

    public async Task<LatencyExecutionResponse> ConfigureExecutionAsync(
        string mode, string modelFolder, int deviceIndex = 0)
    {
        var requestedRunId = Guid.NewGuid().ToString("N");
        _runId = null;

        using var response = await PostAsync(
            "/AiService/LatencyExecutionMode",
            new
            {
                Mode = mode,
                ModelFolder = modelFolder,
                DeviceIndex = deviceIndex,
                RunId = requestedRunId
            });

        var json = await response.Content.ReadAsStringAsync();
        var execution = JsonConvert.DeserializeObject<LatencyExecutionResponse>(json)
               ?? throw new InvalidOperationException(
                   "Invalid latency execution-mode response.");
        if (execution.BackendRevision != RequiredBackendRevision || string.IsNullOrWhiteSpace(execution.RunId))
            throw new InvalidOperationException(
                $"Backend/UI version mismatch. Expected '{RequiredBackendRevision}'. " +
                "Restart the backend from the same project build.");
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
        try
        {
            await ThrowBackendErrorAsync(response, uri);
            return response;
        }
        catch
        {
            response.Dispose();
            throw;
        }
    }

    private static async Task ThrowBackendErrorAsync(HttpResponseMessage response, string uri)
    {
        if (response.IsSuccessStatusCode)
            return;

        var message = (await response.Content.ReadAsStringAsync()).Trim();
        if (string.IsNullOrWhiteSpace(message))
            message = "The backend returned no error details.";

        throw new HttpRequestException(
            $"{uri} returned HTTP {(int)response.StatusCode} ({response.ReasonPhrase}). {message}");
    }
}
