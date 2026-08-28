using System.Threading.Tasks;
using System;
using MaterialDesignThemes.Wpf;
using Newtonsoft.Json;
using Microsoft.WindowsAPICodePack.Dialogs;
using Services.AiService;
using Services.AiService.Interpreter;
using Services.AiService.Responses;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Windows.Input;
using System.Windows;

namespace UI.View;

public partial class LatencyTestWindow : Window
{
    private static readonly HttpClient LatencyClient = new()
    {
        BaseAddress = new Uri($"http://{Services.Configuration.AiServiceUrl}/"),
        Timeout = TimeSpan.FromMinutes(5)
    };

    public static ICommand OpenCommand { get; } = new OpenLatencyTestCommand();

    private string? _latencyDatasetPath;
    private readonly List<LatencyCase> _latencyCases = new();
    private readonly List<ViewLatencyMeasurement> _frontalLatencyMeasurements = new();
    private readonly List<ViewLatencyMeasurement> _lateralLatencyMeasurements = new();
    private int _modelCount;

    public LatencyTestWindow()
    {
        InitializeComponent();

        FrontalLatencyGrid.ItemsSource = _frontalLatencyMeasurements;
        LateralLatencyGrid.ItemsSource = _lateralLatencyMeasurements;
        ResetViewLatencySummaries();
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
        _frontalLatencyMeasurements.Clear();
        _lateralLatencyMeasurements.Clear();
        _modelCount = 0;

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
                    BackendInferenceMilliseconds = "-",
                    ExecutionProvider = "-",
                    TimingDevice = "-",
                    TimingMethod = "-",
                    Status = "Ready"
                });
        }

        LatencyResultsGrid.ItemsSource = null;
        LatencyResultsGrid.ItemsSource = _latencyCases;
        FrontalLatencyGrid.Items.Refresh();
        LateralLatencyGrid.Items.Refresh();

        LatencyCaseCountText.Text = _latencyCases.Count.ToString();
        TotalSelectedSequenceText.Text = (_latencyCases.Count * 2).ToString();
        LatencyAverageText.Text = "- ms";
        LatencyMinText.Text = "- ms";
        LatencyMaxText.Text = "- ms";
        EndToEndAverageText.Text = "- ms";
        EndToEndMinText.Text = "- ms";
        EndToEndMaxText.Text = "- ms";
        ExecutionSummaryText.Text = "-";
        TimingMethodSummaryText.Text = "Timing method: -";
        LatencyProgressBar.Value = 0;
        ResetViewLatencySummaries();

        if (_latencyCases.Count > 0)
        {
            LatencyProgressText.Text =
                $"{_latencyCases.Count} paired cases found and ready " +
                $"({_latencyCases.Count} frontal + {_latencyCases.Count} lateral sequences).";
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

        var models = Directory
            .GetFiles(frontalModelFolder)
            .OrderBy(path => Path.GetFileName(path), System.StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (models.Length == 0)
        {
            MessageBox.Show(
                "No model files were found in the frontal model directory.",
                "Latency Test",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return;
        }

        _modelCount = models.Length;

        StartLatencyTestButton.IsEnabled = false;
        LatencyProgressBar.Value = 0;
        LatencyAverageText.Text = "- ms";
        LatencyMinText.Text = "- ms";
        LatencyMaxText.Text = "- ms";
        EndToEndAverageText.Text = "- ms";
        EndToEndMinText.Text = "- ms";
        EndToEndMaxText.Text = "- ms";
        ExecutionSummaryText.Text = "-";
        TimingMethodSummaryText.Text = "Timing method: -";

        _frontalLatencyMeasurements.Clear();
        _lateralLatencyMeasurements.Clear();
        FrontalLatencyGrid.Items.Refresh();
        LateralLatencyGrid.Items.Refresh();
        ResetViewLatencySummaries();

        FrontalLatencyStatusText.Text =
            "Collecting frontal per-case / per-fold GPU-safe backend timings...";
        LateralLatencyStatusText.Text =
            "Collecting lateral per-case / per-fold GPU-safe backend timings...";

        foreach (var latencyCase in _latencyCases)
        {
            latencyCase.Classification = "-";
            latencyCase.LatencyMilliseconds = "-";
            latencyCase.LatencyValueMilliseconds = null;
            latencyCase.BackendInferenceMilliseconds = "-";
            latencyCase.BackendInferenceValueMilliseconds = null;
            latencyCase.ExecutionProvider = "-";
            latencyCase.TimingDevice = "-";
            latencyCase.TimingMethod = "-";
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
                // Prepare the current image pair once. This stays outside the case
                // stopwatch, preserving the existing classification-latency definition.
                await AiServiceCommunication.LoadImages(
                    latencyCase.FrontalPath,
                    latencyCase.LateralPath);

                var caseStopwatch = Stopwatch.StartNew();
                var responses = new List<ClassificationResponse>();
                var backendInferenceTotalMilliseconds = 0.0;

                foreach (var model in models)
                {
                    var modelName = Path.GetFileName(model);

                    var response =
                        await ClassifySequenceWithTiming(
                            modelName,
                            modelName,
                            latencyCase.FrontalPath,
                            latencyCase.LateralPath);

                    responses.Add(response);
                    backendInferenceTotalMilliseconds += response.InferenceMilliseconds;

                    var executionProvider = NormalizeExecutionProvider(response);
                    var timingDevice = string.IsNullOrWhiteSpace(response.TimingDevice)
                        ? "unknown"
                        : response.TimingDevice;
                    var timingMethod = string.IsNullOrWhiteSpace(response.TimingMethod)
                        ? "unknown"
                        : response.TimingMethod;

                    // All folds in one case are expected to use the same execution provider.
                    // Store it on the case as well so the primary result clearly shows CPU/GPU.
                    latencyCase.ExecutionProvider = executionProvider;
                    latencyCase.TimingDevice = timingDevice;
                    latencyCase.TimingMethod = timingMethod;

                    _frontalLatencyMeasurements.Add(
                        new ViewLatencyMeasurement
                        {
                            CaseName = latencyCase.CaseName,
                            ModelName = modelName,
                            LatencyMilliseconds = response.FrontalInferenceMilliseconds,
                            ExecutionProvider = executionProvider,
                            TimingDevice = timingDevice,
                            TimingMethod = timingMethod,
                            Status = "Complete"
                        });

                    _lateralLatencyMeasurements.Add(
                        new ViewLatencyMeasurement
                        {
                            CaseName = latencyCase.CaseName,
                            ModelName = modelName,
                            LatencyMilliseconds = response.LateralInferenceMilliseconds,
                            ExecutionProvider = executionProvider,
                            TimingDevice = timingDevice,
                            TimingMethod = timingMethod,
                            Status = "Complete"
                        });

                    FrontalLatencyGrid.Items.Refresh();
                    LateralLatencyGrid.Items.Refresh();
                    UpdateViewLatencySummaries();
                }

                var averages =
                    ResultInterpreter.CalculateCombinedResult(responses);

                var resultInterpreter = new ResultInterpreter
                {
                    Threshold = vm.AiClassificationThreshold
                };

                var hasThrombus =
                    resultInterpreter.HasThrombus(averages.Item1);

                caseStopwatch.Stop();
                var elapsedMilliseconds =
                    caseStopwatch.Elapsed.TotalMilliseconds;

                latencyCase.LatencyValueMilliseconds = elapsedMilliseconds;
                latencyCase.LatencyMilliseconds = $"{elapsedMilliseconds:F2}";
                latencyCase.BackendInferenceValueMilliseconds = backendInferenceTotalMilliseconds;
                latencyCase.BackendInferenceMilliseconds = $"{backendInferenceTotalMilliseconds:F2}";
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
                latencyCase.BackendInferenceMilliseconds = "-";
                latencyCase.BackendInferenceValueMilliseconds = null;
                latencyCase.ExecutionProvider = "-";
                latencyCase.TimingDevice = "-";
                latencyCase.TimingMethod = "-";
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

        // Primary scientific result: pure backend ensemble inference across all folds.
        var successfulInferenceLatencies = _latencyCases
            .Where(c => c.BackendInferenceValueMilliseconds.HasValue)
            .Select(c => c.BackendInferenceValueMilliseconds!.Value)
            .ToList();

        if (successfulInferenceLatencies.Count > 0)
        {
            LatencyAverageText.Text =
                $"{successfulInferenceLatencies.Average():F2} ms";
            LatencyMinText.Text =
                $"{successfulInferenceLatencies.Min():F2} ms";
            LatencyMaxText.Text =
                $"{successfulInferenceLatencies.Max():F2} ms";
        }
        else
        {
            LatencyAverageText.Text = "- ms";
            LatencyMinText.Text = "- ms";
            LatencyMaxText.Text = "- ms";
        }

        // Secondary engineering result: complete client/server classification path.
        var successfulEndToEndLatencies = _latencyCases
            .Where(c => c.LatencyValueMilliseconds.HasValue)
            .Select(c => c.LatencyValueMilliseconds!.Value)
            .ToList();

        if (successfulEndToEndLatencies.Count > 0)
        {
            EndToEndAverageText.Text =
                $"{successfulEndToEndLatencies.Average():F2} ms";
            EndToEndMinText.Text =
                $"{successfulEndToEndLatencies.Min():F2} ms";
            EndToEndMaxText.Text =
                $"{successfulEndToEndLatencies.Max():F2} ms";
        }
        else
        {
            EndToEndAverageText.Text = "- ms";
            EndToEndMinText.Text = "- ms";
            EndToEndMaxText.Text = "- ms";
        }

        var executionDescriptions = _latencyCases
            .Where(c => c.BackendInferenceValueMilliseconds.HasValue)
            .Select(c => $"{c.ExecutionProvider} ({c.TimingDevice})")
            .Distinct(System.StringComparer.OrdinalIgnoreCase)
            .ToList();

        ExecutionSummaryText.Text = executionDescriptions.Count > 0
            ? string.Join("; ", executionDescriptions)
            : "-";

        var timingMethods = _latencyCases
            .Where(c => c.BackendInferenceValueMilliseconds.HasValue)
            .Select(c => c.TimingMethod)
            .Where(method => !string.IsNullOrWhiteSpace(method) && method != "-")
            .Distinct(System.StringComparer.OrdinalIgnoreCase)
            .ToList();

        TimingMethodSummaryText.Text = timingMethods.Count > 0
            ? $"Timing method: {string.Join("; ", timingMethods)}"
            : "Timing method: -";

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

        UpdateViewLatencySummaries();
        UpdateViewStatusTexts();

        StartLatencyTestButton.IsEnabled = true;
        StartLatencyTestButton.ToolTip = "Run latency test again";
    }

    private static async Task<LatencyClassificationResponse> ClassifySequenceWithTiming(
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
        using var data = new StringContent(json, Encoding.UTF8, "application/json");
        using var response = await LatencyClient.PostAsync(
            "/AiService/LatencyClassification",
            data);

        response.EnsureSuccessStatusCode();
        var content = await response.Content.ReadAsStringAsync();

        return JsonConvert.DeserializeObject<LatencyClassificationResponse>(content)
               ?? throw new InvalidOperationException(
                   "Failed to convert latency classification response");
    }

    private static string NormalizeExecutionProvider(LatencyClassificationResponse response)
    {
        if (!string.IsNullOrWhiteSpace(response.ExecutionProvider))
        {
            return response.ExecutionProvider.ToUpperInvariant();
        }

        if (!string.IsNullOrWhiteSpace(response.TimingDevice) &&
            response.TimingDevice.StartsWith(
                "cuda",
                System.StringComparison.OrdinalIgnoreCase))
        {
            return "GPU";
        }

        return "CPU";
    }

    private void UpdateViewLatencySummaries()
    {
        UpdateSingleViewSummary(
            _frontalLatencyMeasurements,
            FrontalSequenceCountText,
            FrontalMeasurementCountText,
            FrontalMinText,
            FrontalAverageText,
            FrontalMaxText);

        UpdateSingleViewSummary(
            _lateralLatencyMeasurements,
            LateralSequenceCountText,
            LateralMeasurementCountText,
            LateralMinText,
            LateralAverageText,
            LateralMaxText);

        TotalSelectedSequenceText.Text = (_latencyCases.Count * 2).ToString();
        TotalModelCountText.Text = _modelCount > 0 ? _modelCount.ToString() : "-";
        TotalViewMeasurementText.Text =
            (_frontalLatencyMeasurements.Count + _lateralLatencyMeasurements.Count).ToString();
    }

    private static void UpdateSingleViewSummary(
        IReadOnlyCollection<ViewLatencyMeasurement> measurements,
        System.Windows.Controls.TextBlock sequenceCountText,
        System.Windows.Controls.TextBlock measurementCountText,
        System.Windows.Controls.TextBlock minText,
        System.Windows.Controls.TextBlock averageText,
        System.Windows.Controls.TextBlock maxText)
    {
        sequenceCountText.Text = measurements
            .Select(m => m.CaseName)
            .Distinct(System.StringComparer.OrdinalIgnoreCase)
            .Count()
            .ToString();

        measurementCountText.Text = measurements.Count.ToString();

        if (measurements.Count == 0)
        {
            minText.Text = "- ms";
            averageText.Text = "- ms";
            maxText.Text = "- ms";
            return;
        }

        minText.Text = $"{measurements.Min(m => m.LatencyMilliseconds):F2} ms";
        averageText.Text = $"{measurements.Average(m => m.LatencyMilliseconds):F2} ms";
        maxText.Text = $"{measurements.Max(m => m.LatencyMilliseconds):F2} ms";
    }

    private void UpdateViewStatusTexts()
    {
        FrontalLatencyStatusText.Text = BuildViewStatusText(
            "Frontal",
            _frontalLatencyMeasurements);

        LateralLatencyStatusText.Text = BuildViewStatusText(
            "Lateral",
            _lateralLatencyMeasurements);
    }

    private static string BuildViewStatusText(
        string viewName,
        IReadOnlyCollection<ViewLatencyMeasurement> measurements)
    {
        if (measurements.Count == 0)
        {
            return $"No successful {viewName.ToLowerInvariant()} fold measurements were collected.";
        }

        var executionDescription = string.Join(
            "; ",
            measurements
                .Select(m => $"{m.ExecutionProvider} ({m.TimingDevice}) - {m.TimingMethod}")
                .Distinct());

        var sequenceCount = measurements
            .Select(m => m.CaseName)
            .Distinct(System.StringComparer.OrdinalIgnoreCase)
            .Count();

        var foldCount = measurements
            .Select(m => m.ModelName)
            .Distinct(System.StringComparer.OrdinalIgnoreCase)
            .Count();

        return
            $"{viewName}: {sequenceCount} sequence(s), {foldCount} fold(s), " +
            $"{measurements.Count} successful fold inference measurement(s). " +
            $"Execution: {executionDescription}.";
    }

    private void ResetViewLatencySummaries()
    {
        FrontalSequenceCountText.Text = "0";
        FrontalMeasurementCountText.Text = "0";
        FrontalMinText.Text = "- ms";
        FrontalAverageText.Text = "- ms";
        FrontalMaxText.Text = "- ms";

        LateralSequenceCountText.Text = "0";
        LateralMeasurementCountText.Text = "0";
        LateralMinText.Text = "- ms";
        LateralAverageText.Text = "- ms";
        LateralMaxText.Text = "- ms";

        TotalModelCountText.Text = _modelCount > 0 ? _modelCount.ToString() : "-";
        TotalViewMeasurementText.Text = "0";

        FrontalLatencyStatusText.Text =
            "Run the test to collect frontal per-case / per-fold timings.";
        LateralLatencyStatusText.Text =
            "Run the test to collect lateral per-case / per-fold timings.";
    }

    private sealed class LatencyClassificationResponse : ClassificationResponse
    {
        public double FrontalInferenceMilliseconds { get; set; }
        public double LateralInferenceMilliseconds { get; set; }
        public double InferenceMilliseconds { get; set; }
        public string? TimingDevice { get; set; }
        public string? ExecutionProvider { get; set; }
        public string? TimingMethod { get; set; }
    }

    private sealed class OpenLatencyTestCommand : ICommand
    {
        public bool CanExecute(object? parameter) => parameter is Window;

        public void Execute(object? parameter)
        {
            if (parameter is not Window owner)
            {
                return;
            }

            var latencyWindow = new LatencyTestWindow
            {
                Owner = owner,
                DataContext = owner.DataContext
            };

            latencyWindow.ShowDialog();
        }

        public event EventHandler? CanExecuteChanged
        {
            add { }
            remove { }
        }
    }

    private sealed class LatencyCase
    {
        public string CaseName { get; set; } = "";
        public string FrontalPath { get; set; } = "";
        public string LateralPath { get; set; } = "";
        public string Classification { get; set; } = "";
        public string LatencyMilliseconds { get; set; } = "";
        public double? LatencyValueMilliseconds { get; set; }
        public string BackendInferenceMilliseconds { get; set; } = "";
        public double? BackendInferenceValueMilliseconds { get; set; }
        public string ExecutionProvider { get; set; } = "";
        public string TimingDevice { get; set; } = "";
        public string TimingMethod { get; set; } = "";
        public string Status { get; set; } = "";
    }

    private sealed class ViewLatencyMeasurement
    {
        public string CaseName { get; set; } = "";
        public string ModelName { get; set; } = "";
        public double LatencyMilliseconds { get; set; }
        public string ExecutionProvider { get; set; } = "";
        public string TimingDevice { get; set; } = "";
        public string TimingMethod { get; set; } = "";
        public string Status { get; set; } = "";
    }
}
