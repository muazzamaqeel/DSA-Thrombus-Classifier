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

        var json = JsonConvert.SerializeObject(request);
        using var data =
            new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await Client.PostAsync(
            "/AiService/LatencyClassification",
            data);

        response.EnsureSuccessStatusCode();

        var content = await response.Content.ReadAsStringAsync();

        return JsonConvert.DeserializeObject<LatencyClassificationResponse>(
                   content)
               ?? throw new InvalidOperationException(
                   "Failed to convert latency classification response");
    }
}
