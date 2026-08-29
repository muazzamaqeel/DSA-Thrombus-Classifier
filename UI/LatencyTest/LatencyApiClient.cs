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
        BaseAddress = new Uri(
            $"http://{Services.Configuration.AiServiceUrl}/"),
        Timeout = TimeSpan.FromMinutes(5)
    };

    public async Task ConfigureExecutionUnitAsync(
        string executionUnit)
    {
        var request = new
        {
            ExecutionUnit = executionUnit
        };

        using var response = await PostAsync(
            request,
            "/AiService/LatencyExecutionUnit");
    }

    public async Task PrepareImagesAsync(
        string fileFrontal,
        string fileLateral)
    {
        var request = new
        {
            PathFrontal = fileFrontal,
            PathLateral = fileLateral
        };

        using var response = await PostAsync(
            request,
            "/AiService/LatencyPrepareImages");
    }

    public async Task ReleasePreparedImagesAsync(
        string fileFrontal,
        string fileLateral)
    {
        var request = new
        {
            PathFrontal = fileFrontal,
            PathLateral = fileLateral
        };

        using var response = await PostAsync(
            request,
            "/AiService/LatencyReleaseImages");
    }

    public async Task<LatencyClassificationResponse> ClassifyAsync(
        string modelFrontal,
        string modelLateral,
        string fileFrontal,
        string fileLateral)
    {
        var request = new
        {
            PathFrontal = fileFrontal,
            PathLateral = fileLateral,
            ModelFrontal = modelFrontal,
            ModelLateral = modelLateral
        };

        using var response = await PostAsync(
            request,
            "/AiService/LatencyClassification");

        var content = await response.Content.ReadAsStringAsync();

        return JsonConvert.DeserializeObject<LatencyClassificationResponse>(
                   content)
               ?? throw new InvalidOperationException(
                   "Failed to convert latency classification response");
    }

    private static async Task<HttpResponseMessage> PostAsync(
        object request,
        string uri)
    {
        var json = JsonConvert.SerializeObject(request);
        using var data =
            new StringContent(json, Encoding.UTF8, "application/json");

        var response = await Client.PostAsync(uri, data);
        response.EnsureSuccessStatusCode();

        return response;
    }
}
