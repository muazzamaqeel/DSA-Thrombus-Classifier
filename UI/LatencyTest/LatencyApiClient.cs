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

    public async Task<LatencyExecutionResponse> ConfigureExecutionAsync(
        string mode, string modelFolder)
    {
        using var response = await PostAsync(
            "/AiService/LatencyExecutionMode",
            new { Mode = mode, ModelFolder = modelFolder });

        var json = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<LatencyExecutionResponse>(json)
               ?? throw new InvalidOperationException(
                   "Invalid latency execution-mode response.");
    }

    public Task PrepareImagesAsync(string frontal, string lateral) =>
        PostNoContentAsync("/AiService/LatencyPrepareImages", new
        {
            PathFrontal = frontal,
            PathLateral = lateral
        });

    public Task ReleasePreparedImagesAsync(string frontal, string lateral) =>
        PostNoContentAsync("/AiService/LatencyReleaseImages", new
        {
            PathFrontal = frontal,
            PathLateral = lateral
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
                PathLateral = lateral
            });

        var json = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<LatencyClassificationResponse>(json)
               ?? throw new InvalidOperationException(
                   "Invalid latency classification response.");
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
        response.EnsureSuccessStatusCode();
        return response;
    }
}
