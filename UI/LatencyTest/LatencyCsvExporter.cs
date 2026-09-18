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

        var exportCulture = CultureInfo.CurrentCulture;
        var separator = exportCulture.TextInfo.ListSeparator;
        if (string.IsNullOrWhiteSpace(separator))
            separator = ",";

        using var writer = new StreamWriter(fileName, false, new UTF8Encoding(true));
        writer.NewLine = "\r\n";

        // Excel reads this line and opens every field in its own column.
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
            "Frontal path",
            "Lateral path",
            "Timing method",
            "Model residency",
            "Client case processing (ms; excludes release)",
            "Preparation details JSON",
            "Environment JSON",
            "Ensemble forward (ms)"
        };

        foreach (var model in modelNames)
        {
            header.Add($"Frontal {model} output");
            header.Add($"Lateral {model} output");
            header.Add($"Frontal {model} classification (ms)");
            header.Add($"Lateral {model} classification (ms)");
            header.Add($"Frontal {model} benchmark JSON");
            header.Add($"Lateral {model} benchmark JSON");
            header.Add($"Frontal {model} forward (ms)");
            header.Add($"Lateral {model} forward (ms)");
        }

        header.AddRange(new[]
        {
            "Completed paired cases",
            "Mean classification (ms)",
            "Sample SD classification (ms; n-1)"
        });

        var summary = LatencyStatistics.Calculate(
            cases.Where(x => x.Status == "Complete" && x.InferenceMilliseconds.HasValue)
                .Select(x => x.InferenceMilliseconds!.Value));

        WriteRow(writer, header, separator);

        foreach (var item in cases)
        {
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
                item.FrontalPath,
                item.LateralPath,
                item.TimingMethod,
                item.ModelResidency,
                Number(item.ClientCaseMilliseconds, exportCulture),
                item.PreparationJson,
                item.EnvironmentJson,
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
                fields.Add(f?.BenchmarkJson ?? "");
                fields.Add(l?.BenchmarkJson ?? "");
                fields.Add(Number(f?.ForwardMilliseconds, exportCulture));
                fields.Add(Number(l?.ForwardMilliseconds, exportCulture));
            }

            fields.Add(summary.Count.ToString(exportCulture));
            fields.Add(Number(summary.Mean, exportCulture));
            fields.Add(Number(summary.StandardDeviation, exportCulture));

            WriteRow(writer, fields, separator);
        }
    }

    private static string Number(double? value, CultureInfo culture) =>
        value?.ToString("G17", culture) ?? "";

    private static void WriteRow(
        TextWriter writer,
        IEnumerable<string> fields,
        string separator) =>
        writer.WriteLine(string.Join(separator, fields.Select(Escape)));

    private static string Escape(string value) =>
        "\"" + value.Replace("\"", "\"\"") + "\"";
}
