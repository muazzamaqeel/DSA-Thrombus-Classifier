using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace UI.LatencyTest;

public sealed class LatencyDatasetScanner
{
    public List<LatencyCase> Scan(string datasetPath)
    {
        var niftiFiles = Directory
            .GetFiles(datasetPath, "*.nii.gz", SearchOption.TopDirectoryOnly)
            .OrderBy(path => path)
            .ToList();

        var frontalFiles = niftiFiles
            .Where(path => Path.GetFileName(path).Contains("_C_"))
            .ToList();

        var lateralFiles = niftiFiles
            .Where(path => Path.GetFileName(path).Contains("_S_"))
            .ToList();

        var cases = new List<LatencyCase>();

        foreach (var frontalPath in frontalFiles)
        {
            var frontalFileName =
                Path.GetFileName(frontalPath) ?? string.Empty;

            var expectedLateralFileName =
                frontalFileName.Replace("_C_", "_S_");

            var lateralPath = lateralFiles.FirstOrDefault(
                path => string.Equals(
                    Path.GetFileName(path),
                    expectedLateralFileName,
                    StringComparison.OrdinalIgnoreCase));

            if (lateralPath == null)
            {
                continue;
            }

            var markerPosition = frontalFileName.IndexOf(
                "_C_",
                StringComparison.Ordinal);

            var caseName = markerPosition > 0
                ? frontalFileName[..markerPosition]
                : frontalFileName;

            cases.Add(
                new LatencyCase
                {
                    CaseName = caseName,
                    FrontalPath = frontalPath,
                    LateralPath = lateralPath,
                    Classification = "-",
                    LatencyMilliseconds = "-",
                    BackendInferenceMilliseconds = "-",
                    ExecutionProvider = "-",
                    TimingDevice = "-",
                    TimingMethod = "-",
                    Status = "Ready"
                });
        }

        return cases;
    }
}
