using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace UI.LatencyTest;

public static class LatencyCsvExporter
{
    public static void Write(
        string fileName,
        IReadOnlyCollection<LatencyCase> cases,
        IReadOnlyCollection<ViewLatencyMeasurement> frontal,
        IReadOnlyCollection<ViewLatencyMeasurement> lateral,
        IReadOnlyCollection<string> modelNames)
    {
        var frontalByCase = frontal.ToDictionary(x => (x.CaseKey, x.ModelName));
        var lateralByCase = lateral.ToDictionary(x => (x.CaseKey, x.ModelName));

        var exportCulture = CultureInfo.CurrentCulture;
        var separator = exportCulture.TextInfo.ListSeparator;
        if (string.IsNullOrWhiteSpace(separator))
            separator = ",";

        using var writer = new StreamWriter(fileName, false, new UTF8Encoding(true));
        writer.NewLine = "\r\n";

        // Makes Excel open the CSV with the intended delimiter immediately.
        writer.WriteLine($"sep={separator}");

        var header = new List<string>
        {
            "Case",
            "Classification",
            "Model output",
            "Threshold",
            "Ensemble classification (ms)",
            "Execution provider",
            "Timing device",
            "Status",
            "Frontal frames",
            "Lateral frames",
            "Frontal path",
            "Lateral path",
            "Ensemble forward (ms)"
        };

        foreach (var model in modelNames)
        {
            header.Add($"Frontal {model} output");
            header.Add($"Lateral {model} output");
            header.Add($"Frontal {model} classification (ms)");
            header.Add($"Lateral {model} classification (ms)");
            header.Add($"Frontal {model} forward (ms)");
            header.Add($"Lateral {model} forward (ms)");
        }

        WriteRow(writer, header, separator);

        foreach (var item in cases)
        {
            var frames = GetFrameCounts(item.PreparationJson);

            var fields = new List<string>
            {
                item.CaseName,
                item.Classification,
                Number(item.ModelOutput, exportCulture),
                Number(item.Threshold, exportCulture),
                Number(item.InferenceMilliseconds, exportCulture),
                item.ExecutionProvider,
                item.TimingDevice,
                item.Status,
                Integer(frames.Frontal, exportCulture),
                Integer(frames.Lateral, exportCulture),
                item.FrontalPath,
                item.LateralPath,
                Number(item.ForwardMilliseconds, exportCulture)
            };

            foreach (var model in modelNames)
            {
                frontalByCase.TryGetValue((item.FrontalPath, model), out var f);
                lateralByCase.TryGetValue((item.FrontalPath, model), out var l);

                fields.Add(Number(f?.ModelOutput, exportCulture));
                fields.Add(Number(l?.ModelOutput, exportCulture));
                fields.Add(Number(f?.LatencyMilliseconds, exportCulture));
                fields.Add(Number(l?.LatencyMilliseconds, exportCulture));
                fields.Add(Number(f?.ForwardMilliseconds, exportCulture));
                fields.Add(Number(l?.ForwardMilliseconds, exportCulture));
            }

            WriteRow(writer, fields, separator);
        }

        // Keep the spreadsheet readable: one compact summary block below the case table.
        var completed = cases
            .Where(x => x.Status == "Complete" && x.InferenceMilliseconds.HasValue)
            .ToList();

        var latencySummary = LatencyStatistics.Calculate(
            completed.Select(x => x.InferenceMilliseconds!.Value));

        var testedFrameCounts = completed
            .SelectMany(item =>
            {
                var frames = GetFrameCounts(item.PreparationJson);
                return new int?[] { frames.Frontal, frames.Lateral };
            })
            .Where(value => value.HasValue)
            .Select(value => (double)value!.Value)
            .ToList();

        double? averageFrames = testedFrameCounts.Count == 0
            ? null
            : testedFrameCounts.Average();

        writer.WriteLine();
        WriteRow(writer, new[] { "Summary metric", "Value" }, separator);
        WriteRow(writer, new[]
        {
            "Mean classification (ms)",
            Number(latencySummary.Mean, exportCulture)
        }, separator);
        WriteRow(writer, new[]
        {
            "Min classification (ms)",
            Number(latencySummary.Min, exportCulture)
        }, separator);
        WriteRow(writer, new[]
        {
            "Max classification (ms)",
            Number(latencySummary.Max, exportCulture)
        }, separator);
        WriteRow(writer, new[]
        {
            "Average frames per tested sequence",
            Number(averageFrames, exportCulture)
        }, separator);
    }

    private static (int? Frontal, int? Lateral) GetFrameCounts(string preparationJson)
    {
        if (string.IsNullOrWhiteSpace(preparationJson))
            return (null, null);

        try
        {
            using var document = JsonDocument.Parse(preparationJson);
            var root = document.RootElement;

            return (
                ReadFrameCount(root, "FrontalShape"),
                ReadFrameCount(root, "LateralShape")
            );
        }
        catch (JsonException)
        {
            return (null, null);
        }
    }

    private static int? ReadFrameCount(JsonElement root, string propertyName)
    {
        if (!root.TryGetProperty(propertyName, out var shape) ||
            shape.ValueKind != JsonValueKind.Array ||
            shape.GetArrayLength() < 2)
            return null;

        var frameElement = shape[1];
        return frameElement.ValueKind == JsonValueKind.Number &&
               frameElement.TryGetInt32(out var frameCount)
            ? frameCount
            : null;
    }

    private static string Number(double? value, CultureInfo culture) =>
        value?.ToString("G17", culture) ?? "";

    private static string Integer(int? value, CultureInfo culture) =>
        value?.ToString(culture) ?? "";

    private static void WriteRow(
        TextWriter writer,
        IEnumerable<string> fields,
        string separator) =>
        writer.WriteLine(string.Join(separator, fields.Select(Escape)));

    private static string Escape(string value) =>
        "\"" + value.Replace("\"", "\"\"") + "\"";
}
