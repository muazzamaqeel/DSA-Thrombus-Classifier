using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

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

        // UTF-8 BOM keeps non-ASCII case names readable when opening in Excel.
        using var writer = new StreamWriter(fileName, false, new UTF8Encoding(true));
        writer.NewLine = "\r\n";

        var header = new List<string>
        {
            "Case", "Classification", "Model output", "Threshold",
            "Ensemble forward (ms)", "Execution provider", "Timing device", "Status",
            "Frontal path", "Lateral path", "Timing method", "Model residency",
            "Client case processing (ms; excludes release)", "Preparation details JSON", "Environment JSON"
        };
        foreach (var model in modelNames)
        {
            header.Add($"Frontal {model} output");
            header.Add($"Lateral {model} output");
            header.Add($"Frontal {model} forward (ms)");
            header.Add($"Lateral {model} forward (ms)");
            header.Add($"Frontal {model} benchmark JSON");
            header.Add($"Lateral {model} benchmark JSON");
        }
        WriteRow(writer, header);

        foreach (var item in cases)
        {
            var fields = new List<string>
            {
                item.CaseName, item.Classification, Number(item.ModelOutput),
                Number(item.Threshold), Number(item.InferenceMilliseconds),
                item.ExecutionProvider, item.TimingDevice, item.Status,
                item.FrontalPath, item.LateralPath, item.TimingMethod, item.ModelResidency,
                Number(item.ClientCaseMilliseconds), item.PreparationJson, item.EnvironmentJson
            };
            foreach (var model in modelNames)
            {
                frontalByCase.TryGetValue((item.FrontalPath, model), out var f);
                lateralByCase.TryGetValue((item.FrontalPath, model), out var l);
                fields.Add(Number(f?.ModelOutput));
                fields.Add(Number(l?.ModelOutput));
                fields.Add(Number(f?.LatencyMilliseconds));
                fields.Add(Number(l?.LatencyMilliseconds));
                fields.Add(f?.BenchmarkJson ?? "");
                fields.Add(l?.BenchmarkJson ?? "");
            }
            WriteRow(writer, fields);
        }
    }

    // Preserve the stored double precision; never format scores as F2 or percent.
    private static string Number(double? value) =>
        value?.ToString("G17", CultureInfo.InvariantCulture) ?? "";

    private static void WriteRow(TextWriter writer, IEnumerable<string> fields) =>
        writer.WriteLine(string.Join(",", fields.Select(Escape)));

    private static string Escape(string value) =>
        "\"" + value.Replace("\"", "\"\"") + "\"";
}
