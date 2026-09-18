using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace UI.LatencyTest;

public sealed class LatencyDatasetScanner
{
    public List<LatencyCase> Scan(string datasetPath)
    {
        if (string.IsNullOrWhiteSpace(datasetPath))
            throw new ArgumentException("Dataset path is empty.", nameof(datasetPath));

        datasetPath = Path.GetFullPath(datasetPath);
        if (!Directory.Exists(datasetPath))
            throw new DirectoryNotFoundException($"Dataset folder does not exist: {datasetPath}");

        // Support both the original flat *.nii.gz dataset and recursive datasets
        // containing *.nii / *.nii.gz files in arbitrary subdirectories.
        var files = Directory.EnumerateFiles(datasetPath, "*", SearchOption.AllDirectories)
            .Where(IsNiftiFile)
            .OrderBy(path => Path.GetRelativePath(datasetPath, path), StringComparer.OrdinalIgnoreCase)
            .ToList();

        var cases = new List<LatencyCase>();
        var usedFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // 1) Preserve the original _C_ / _S_ filename pairing rule.
        //    It now also works recursively and for both .nii and .nii.gz.
        var filesByFullPath = files.ToDictionary(
            Path.GetFullPath,
            path => path,
            StringComparer.OrdinalIgnoreCase);

        foreach (var frontalPath in files.Where(path =>
                     Path.GetFileName(path).Contains("_C_", StringComparison.OrdinalIgnoreCase)))
        {
            var frontalName = Path.GetFileName(frontalPath);
            var lateralName = frontalName.Replace(
                "_C_", "_S_", StringComparison.OrdinalIgnoreCase);
            var directory = Path.GetDirectoryName(frontalPath) ?? datasetPath;
            var expectedLateral = Path.GetFullPath(Path.Combine(directory, lateralName));

            if (!filesByFullPath.TryGetValue(expectedLateral, out var lateralPath))
                continue;

            cases.Add(new LatencyCase
            {
                CaseName = GetLegacyCaseName(datasetPath, frontalPath),
                FrontalPath = frontalPath,
                LateralPath = lateralPath
            });

            usedFiles.Add(Path.GetFullPath(frontalPath));
            usedFiles.Add(Path.GetFullPath(lateralPath));
        }

        // 2) Support directory-based datasets such as:
        //
        //    <selected root>\November 2024\155-ric_ma_m79\pre frontal\1Cprf.nii
        //    <selected root>\November 2024\155-ric_ma_m79\pre lateral \17prl.nii
        //
        //    Both files become one case named:
        //    November 2024\155-ric_ma_m79\pre
        //
        // The nearest ancestor directory containing "frontal" or "lateral"
        // determines the view. Removing that view word gives the relative case path.
        var structuredGroups = new Dictionary<string, StructuredGroup>(
            StringComparer.OrdinalIgnoreCase);

        foreach (var path in files)
        {
            var fullPath = Path.GetFullPath(path);
            if (usedFiles.Contains(fullPath))
                continue;

            if (!TryGetStructuredCase(
                    datasetPath, path, out var caseName, out var view))
                continue;

            if (!structuredGroups.TryGetValue(caseName, out var group))
            {
                group = new StructuredGroup(caseName);
                structuredGroups.Add(caseName, group);
            }

            if (view == DatasetView.Frontal)
                group.FrontalFiles.Add(path);
            else
                group.LateralFiles.Add(path);
        }

        foreach (var group in structuredGroups.Values
                     .OrderBy(item => item.CaseName, StringComparer.OrdinalIgnoreCase))
        {
            // Never guess when a directory contains several possible files for one
            // view. A wrong frontal/lateral pairing would invalidate the benchmark.
            if (group.FrontalFiles.Count > 1 || group.LateralFiles.Count > 1)
            {
                var frontal = string.Join(", ",
                    group.FrontalFiles.Select(path => Path.GetRelativePath(datasetPath, path)));
                var lateral = string.Join(", ",
                    group.LateralFiles.Select(path => Path.GetRelativePath(datasetPath, path)));

                throw new InvalidDataException(
                    $"Ambiguous NIfTI pair for relative case '{group.CaseName}'. " +
                    $"Expected at most one frontal and one lateral NIfTI file, but found " +
                    $"{group.FrontalFiles.Count} frontal and {group.LateralFiles.Count} lateral. " +
                    $"Frontal: [{frontal}] Lateral: [{lateral}]");
            }

            // As before, incomplete pairs are ignored.
            if (group.FrontalFiles.Count != 1 || group.LateralFiles.Count != 1)
                continue;

            cases.Add(new LatencyCase
            {
                CaseName = group.CaseName,
                FrontalPath = group.FrontalFiles[0],
                LateralPath = group.LateralFiles[0]
            });
        }

        return cases
            .OrderBy(item => item.CaseName, StringComparer.OrdinalIgnoreCase)
            .ThenBy(item => item.FrontalPath, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static bool IsNiftiFile(string path) =>
        path.EndsWith(".nii", StringComparison.OrdinalIgnoreCase) ||
        path.EndsWith(".nii.gz", StringComparison.OrdinalIgnoreCase);

    private static string GetLegacyCaseName(string datasetPath, string frontalPath)
    {
        var frontalName = Path.GetFileName(frontalPath);
        var marker = frontalName.IndexOf("_C_", StringComparison.OrdinalIgnoreCase);
        var originalCaseName = marker >= 0 ? frontalName[..marker] : frontalName;

        var directory = Path.GetDirectoryName(frontalPath) ?? datasetPath;
        var relativeDirectory = Path.GetRelativePath(datasetPath, directory);

        return relativeDirectory == "."
            ? originalCaseName
            : NormalizeRelativePath(Path.Combine(relativeDirectory, originalCaseName));
    }

    private static bool TryGetStructuredCase(
        string datasetPath,
        string filePath,
        out string caseName,
        out DatasetView view)
    {
        caseName = "";
        view = default;

        var directory = Path.GetDirectoryName(filePath);
        while (!string.IsNullOrWhiteSpace(directory) &&
               !PathsEqual(directory, datasetPath))
        {
            var folderName = Path.GetFileName(
                directory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));

            if (TryGetView(folderName, out view, out var viewWord))
            {
                var relativeViewDirectory = Path.GetRelativePath(datasetPath, directory);
                var relativeParent = Path.GetDirectoryName(relativeViewDirectory);
                if (relativeParent == ".")
                    relativeParent = "";

                var caseLeaf = RemoveViewWord(folderName, viewWord);

                if (string.IsNullOrWhiteSpace(relativeParent))
                    caseName = string.IsNullOrWhiteSpace(caseLeaf) ? "." : caseLeaf;
                else
                    caseName = string.IsNullOrWhiteSpace(caseLeaf)
                        ? relativeParent
                        : Path.Combine(relativeParent, caseLeaf);

                caseName = NormalizeRelativePath(caseName);
                return true;
            }

            directory = Path.GetDirectoryName(directory);
        }

        return false;
    }

    private static bool TryGetView(
        string folderName,
        out DatasetView view,
        out string viewWord)
    {
        var frontalIndex = folderName.IndexOf(
            "frontal", StringComparison.OrdinalIgnoreCase);
        var lateralIndex = folderName.IndexOf(
            "lateral", StringComparison.OrdinalIgnoreCase);

        if (frontalIndex >= 0 && lateralIndex < 0)
        {
            view = DatasetView.Frontal;
            viewWord = "frontal";
            return true;
        }

        if (lateralIndex >= 0 && frontalIndex < 0)
        {
            view = DatasetView.Lateral;
            viewWord = "lateral";
            return true;
        }

        view = default;
        viewWord = "";
        return false;
    }

    private static string RemoveViewWord(string folderName, string viewWord)
    {
        var index = folderName.IndexOf(viewWord, StringComparison.OrdinalIgnoreCase);
        if (index < 0)
            return folderName;

        var result = folderName.Remove(index, viewWord.Length)
            .Trim(' ', '_', '-', '.');

        while (result.Contains("  ", StringComparison.Ordinal))
            result = result.Replace("  ", " ", StringComparison.Ordinal);

        return result.Trim(' ', '_', '-', '.');
    }

    private static string NormalizeRelativePath(string path) =>
        path.Replace(Path.AltDirectorySeparatorChar, Path.DirectorySeparatorChar);

    private static bool PathsEqual(string left, string right) =>
        string.Equals(
            Path.GetFullPath(left).TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            Path.GetFullPath(right).TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            StringComparison.OrdinalIgnoreCase);

    private enum DatasetView
    {
        Frontal,
        Lateral
    }

    private sealed class StructuredGroup
    {
        public StructuredGroup(string caseName) => CaseName = caseName;

        public string CaseName { get; }
        public List<string> FrontalFiles { get; } = new();
        public List<string> LateralFiles { get; } = new();
    }
}
