using MaterialDesignThemes.Wpf;
using Microsoft.WindowsAPICodePack.Dialogs;
using Services.AiService;
using Services.AiService.Interpreter;
using Services.AiService.Responses;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows;

namespace UI.View;

public partial class LatencyTestWindow : Window
{
    private string? _latencyDatasetPath;
    private readonly List<LatencyCase> _latencyCases = new();

    public LatencyTestWindow()
    {
        InitializeComponent();
    }

    private void SelectLatencyDataset_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new CommonOpenFileDialog
        {
            IsFolderPicker = true,
            Title = "Select latency test dataset folder"
        };

        if (dialog.ShowDialog() != CommonFileDialogResult.Ok)
        {
            return;
        }

        _latencyDatasetPath = dialog.FileName;
        LatencyDatasetPathText.Text = _latencyDatasetPath;
        ScanLatencyDataset(_latencyDatasetPath);
    }

    private void ScanLatencyDataset(string datasetPath)
    {
        _latencyCases.Clear();

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

        foreach (var frontalPath in frontalFiles)
        {
            var frontalFileName = Path.GetFileName(frontalPath);
            var expectedLateralFileName = frontalFileName.Replace("_C_", "_S_");

            var lateralPath = lateralFiles.FirstOrDefault(
                path => string.Equals(
                    Path.GetFileName(path),
                    expectedLateralFileName,
                    System.StringComparison.OrdinalIgnoreCase));

            if (lateralPath == null)
            {
                continue;
            }

            var caseName = frontalFileName;
            var markerPosition = caseName.IndexOf("_C_");

            if (markerPosition > 0)
            {
                caseName = caseName.Substring(0, markerPosition);
            }

            _latencyCases.Add(
                new LatencyCase
                {
                    CaseName = caseName,
                    FrontalPath = frontalPath,
                    LateralPath = lateralPath,
                    Classification = "-",
                    LatencyMilliseconds = "-",
                    Status = "Ready"
                });
        }

        LatencyResultsGrid.ItemsSource = null;
        LatencyResultsGrid.ItemsSource = _latencyCases;
        LatencyCaseCountText.Text = _latencyCases.Count.ToString();
        LatencyAverageText.Text = "- ms";
        LatencyMinText.Text = "- ms";
        LatencyMaxText.Text = "- ms";
        LatencyProgressBar.Value = 0;

        if (_latencyCases.Count > 0)
        {
            LatencyProgressText.Text =
                $"{_latencyCases.Count} paired cases found and ready.";
            StartLatencyTestButton.IsEnabled = true;
            StartLatencyTestButton.ToolTip = "Start latency test";
        }
        else
        {
            LatencyProgressText.Text =
                "No valid frontal/lateral pairs were found.";
            StartLatencyTestButton.IsEnabled = false;
            StartLatencyTestButton.ToolTip =
                "No valid dataset pairs were found.";
        }
    }

    private async void StartLatencyTestButton_Click(
        object sender,
        RoutedEventArgs e)
    {
        if (_latencyCases.Count == 0)
        {
            MessageBox.Show(
                "No dataset cases are available.",
                "Latency Test",
                MessageBoxButton.OK,
                MessageBoxImage.Warning);
            return;
        }

        if (DataContext is not MainWindowViewModel vm)
        {
            MessageBox.Show(
                "The classifier view model is unavailable.",
                "Latency Test",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return;
        }

        if (string.IsNullOrWhiteSpace(vm.ModelSelectionFolder))
        {
            MessageBox.Show(
                "Please select a model folder first.",
                "Latency Test",
                MessageBoxButton.OK,
                MessageBoxImage.Warning);
            return;
        }

        if (vm.ModelSelectionFolderBadge.Kind != PackIconKind.Check)
        {
            MessageBox.Show(
                "The models are not ready yet. Please wait until model initialization has completed successfully.",
                "Latency Test",
                MessageBoxButton.OK,
                MessageBoxImage.Warning);
            return;
        }

        var frontalModelFolder = Path.Join(vm.ModelSelectionFolder, "frontal");

        if (!Directory.Exists(frontalModelFolder))
        {
            MessageBox.Show(
                "The frontal model directory could not be found.",
                "Latency Test",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return;
        }

        var models = Directory.GetFiles(frontalModelFolder);

        if (models.Length == 0)
        {
            MessageBox.Show(
                "No model files were found in the frontal model directory.",
                "Latency Test",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return;
        }

        StartLatencyTestButton.IsEnabled = false;
        LatencyProgressBar.Value = 0;
        LatencyAverageText.Text = "- ms";
        LatencyMinText.Text = "- ms";
        LatencyMaxText.Text = "- ms";

        foreach (var latencyCase in _latencyCases)
        {
            latencyCase.Classification = "-";
            latencyCase.LatencyMilliseconds = "-";
            latencyCase.LatencyValueMilliseconds = null;
            latencyCase.Status = "Ready";
        }

        LatencyResultsGrid.Items.Refresh();
        var completedCases = 0;

        foreach (var latencyCase in _latencyCases)
        {
            latencyCase.Status = "Processing";
            LatencyProgressText.Text =
                $"Processing {completedCases + 1} of {_latencyCases.Count}: " +
                latencyCase.CaseName;
            LatencyResultsGrid.Items.Refresh();

            try
            {
                // The backend requires each image pair to be prepared before classification.
                // Keep this outside the stopwatch so the measured value matches the working
                // latency-test behavior: model classification only, not image preprocessing.
                await AiServiceCommunication.LoadImages(
                    latencyCase.FrontalPath,
                    latencyCase.LateralPath);

                var stopwatch = Stopwatch.StartNew();
                var responses = new List<ClassificationResponse>();

                foreach (var model in models)
                {
                    var modelName = Path.GetFileName(model);
                    var response =
                        await AiServiceCommunication.ClassifySequence(
                            modelName,
                            modelName,
                            latencyCase.FrontalPath,
                            latencyCase.LateralPath);
                    responses.Add(response);
                }

                var averages =
                    ResultInterpreter.CalculateCombinedResult(responses);

                var resultInterpreter = new ResultInterpreter
                {
                    Threshold = vm.AiClassificationThreshold
                };

                var hasThrombus =
                    resultInterpreter.HasThrombus(averages.Item1);

                stopwatch.Stop();
                var elapsedMilliseconds =
                    stopwatch.Elapsed.TotalMilliseconds;

                latencyCase.LatencyValueMilliseconds = elapsedMilliseconds;
                latencyCase.LatencyMilliseconds = $"{elapsedMilliseconds:F2}";
                latencyCase.Classification = hasThrombus
                    ? "Thrombus detected"
                    : "No Thrombus detected";
                latencyCase.Status = "Complete";
            }
            catch (System.Exception exception)
            {
                latencyCase.Classification = "Error";
                latencyCase.LatencyMilliseconds = "-";
                latencyCase.LatencyValueMilliseconds = null;
                latencyCase.Status = "Failed";

                MessageBox.Show(
                    $"Classification failed for case " +
                    $"{latencyCase.CaseName}.\n\n" +
                    $"{exception.Message}",
                    "Latency Test",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
            }

            completedCases++;
            LatencyProgressBar.Value =
                ((double)completedCases / _latencyCases.Count) * 100.0;
            LatencyResultsGrid.Items.Refresh();
        }

        var successfulLatencies = _latencyCases
            .Where(c => c.LatencyValueMilliseconds.HasValue)
            .Select(c => c.LatencyValueMilliseconds!.Value)
            .ToList();

        if (successfulLatencies.Count > 0)
        {
            LatencyAverageText.Text =
                $"{successfulLatencies.Average():F2} ms";
            LatencyMinText.Text =
                $"{successfulLatencies.Min():F2} ms";
            LatencyMaxText.Text =
                $"{successfulLatencies.Max():F2} ms";
        }
        else
        {
            LatencyAverageText.Text = "- ms";
            LatencyMinText.Text = "- ms";
            LatencyMaxText.Text = "- ms";
        }

        var failedCases =
            _latencyCases.Count(c => c.Status == "Failed");

        if (failedCases == 0)
        {
            LatencyProgressText.Text =
                $"Completed {_latencyCases.Count} of " +
                $"{_latencyCases.Count} cases.";
        }
        else
        {
            LatencyProgressText.Text =
                $"Completed {_latencyCases.Count - failedCases} cases. " +
                $"{failedCases} failed.";
        }

        StartLatencyTestButton.IsEnabled = true;
        StartLatencyTestButton.ToolTip = "Run latency test again";
    }

    private sealed class LatencyCase
    {
        public string CaseName { get; set; } = "";
        public string FrontalPath { get; set; } = "";
        public string LateralPath { get; set; } = "";
        public string Classification { get; set; } = "";
        public string LatencyMilliseconds { get; set; } = "";
        public double? LatencyValueMilliseconds { get; set; }
        public string Status { get; set; } = "";
    }
}
