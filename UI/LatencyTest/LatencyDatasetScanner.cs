using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace UI.LatencyTest;

public sealed class LatencyDatasetScanner
{
    public List<LatencyCase> Scan(string datasetPath)
    {
        var files = Directory.GetFiles(
            datasetPath, "*.nii.gz", SearchOption.TopDirectoryOnly);

        var lateralByName = files
            .Where(path => Path.GetFileName(path).Contains("_S_"))
            .ToDictionary(
                path => Path.GetFileName(path),
                path => path,
                StringComparer.OrdinalIgnoreCase);

        return files
            .Where(path => Path.GetFileName(path).Contains("_C_"))
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .Select(frontal => CreateCase(frontal, lateralByName))
            .Where(item => item is not null)
            .Select(item => item!)
            .ToList();
    }

    private static LatencyCase? CreateCase(
        string frontalPath,
        IReadOnlyDictionary<string, string> lateralByName)
    {
        var frontalName = Path.GetFileName(frontalPath);
        var lateralName = frontalName.Replace("_C_", "_S_");

        if (!lateralByName.TryGetValue(lateralName, out var lateralPath))
            return null;

        return new LatencyCase
        {
            CaseName = frontalName[..frontalName.IndexOf(
                "_C_", StringComparison.Ordinal)],
            FrontalPath = frontalPath,
            LateralPath = lateralPath
        };
    }
}
