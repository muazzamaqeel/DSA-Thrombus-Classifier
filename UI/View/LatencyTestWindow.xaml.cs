using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using MaterialDesignThemes.Wpf;
using Microsoft.WindowsAPICodePack.Dialogs;
using UI.LatencyTest;

namespace UI.View;

public partial class LatencyTestWindow : Window
{
    private readonly LatencyDatasetScanner _datasetScanner = new();
    private readonly LatencyTestRunner _runner = new();

    private readonly List<LatencyCase> _latencyCases = new();
    private readonly List<ViewLatencyMeasurement>
        _frontalLatencyMeasurements = new();
    private readonly List<ViewLatencyMeasurement>
        _lateralLatencyMeasurements = new();

    private int _modelCount;

    public static ICommand OpenCommand { get; } =
        new LatencyTestOpenCommand();

    public LatencyTestWindow()
    {
        InitializeComponent();

        LatencyResultsGrid.ItemsSource = _latencyCases;
        FrontalLatencyGrid.ItemsSource =
            _frontalLatencyMeasurements;
        LateralLatencyGrid.ItemsSource =
            _lateralLatencyMeasurements;

        ResetViewLatencySummaries();
    }

    // ---------------------------------------------------------------------
    // 1. CONFIGURE DATASET
    // ---------------------------------------------------------------------

    private void SelectLatencyDataset_Click(
        object sender,
        RoutedEventArgs e)
    {
        var dialog = new CommonOpenFileDialog
        {
            IsFolderPicker = true,
            Title = "Select latency test dataset folder"
        };

        if (dialog.ShowDialog() !=
            CommonFileDialogResult.Ok)
        {
            return;
        }

        LatencyDatasetPathText.Text = dialog.FileName;

        _latencyCases.Clear();
        _latencyCases.AddRange(
            _datasetScanner.Scan(dialog.FileName));

        ResetAfterDatasetScan();
    }

    // ---------------------------------------------------------------------
    // 2. VALIDATE + START
    // ---------------------------------------------------------------------

    private async void StartLatencyTestButton_Click(
        object sender,
        RoutedEventArgs e)
    {
        if (!TryCreateRunConfiguration(
                out var vm,
                out var modelNames))
        {
            return;
        }

        BeginRunUi(modelNames.Count);

        var completedCases = 0;

        // -----------------------------------------------------------------
        // 3. PROCESS CASES
        // -----------------------------------------------------------------
        foreach (var latencyCase in _latencyCases)
        {
            latencyCase.Status = "Processing";
            LatencyProgressText.Text =
                $"Processing {completedCases + 1} of " +
                $"{_latencyCases.Count}: " +
                latencyCase.CaseName;

            LatencyResultsGrid.Items.Refresh();

            try
            {
                var result = await _runner.RunCaseAsync(
                    latencyCase,
                    modelNames,
                    vm.AiClassificationThreshold,
                    OnFoldCompleted);

                ApplySuccessfulCaseResult(
                    latencyCase,
                    result);
            }
            catch (Exception exception)
            {
                ApplyFailedCaseResult(latencyCase);

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
                ((double)completedCases /
                 _latencyCases.Count) * 100.0;

            LatencyResultsGrid.Items.Refresh();
        }

        // -----------------------------------------------------------------
        // 4. SUMMARIZE
        // -----------------------------------------------------------------
        CompleteRunUi();
    }

    // ---------------------------------------------------------------------
    // CONFIGURATION / VALIDATION
    // ---------------------------------------------------------------------

    private bool TryCreateRunConfiguration(
        out MainWindowViewModel vm,
        out List<string> modelNames)
    {
        vm = null!;
        modelNames = new List<string>();

        if (_latencyCases.Count == 0)
        {
            ShowWarning(
                "No dataset cases are available.");
            return false;
        }

        if (DataContext is not MainWindowViewModel viewModel)
        {
            MessageBox.Show(
                "The classifier view model is unavailable.",
                "Latency Test",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return false;
        }

        vm = viewModel;

        if (string.IsNullOrWhiteSpace(
                vm.ModelSelectionFolder))
        {
            ShowWarning(
                "Please select a model folder first.");
            return false;
        }

        if (vm.ModelSelectionFolderBadge.Kind !=
            PackIconKind.Check)
        {
            ShowWarning(
                "The models are not ready yet. " +
                "Please wait until model initialization " +
                "has completed successfully.");
            return false;
        }

        var frontalModelFolder = Path.Join(
            vm.ModelSelectionFolder,
            "frontal");

        if (!Directory.Exists(frontalModelFolder))
        {
            MessageBox.Show(
                "The frontal model directory " +
                "could not be found.",
                "Latency Test",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return false;
        }

        modelNames = Directory
            .GetFiles(frontalModelFolder)
            .OrderBy(
                path => Path.GetFileName(path),
                StringComparer.OrdinalIgnoreCase)
            .Select(path => Path.GetFileName(path))
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Select(name => name!)
            .ToList();

        if (modelNames.Count == 0)
        {
            MessageBox.Show(
                "No model files were found in " +
                "the frontal model directory.",
                "Latency Test",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return false;
        }

        return true;
    }

    private static void ShowWarning(string message)
    {
        MessageBox.Show(
            message,
            "Latency Test",
            MessageBoxButton.OK,
            MessageBoxImage.Warning);
    }

    // ---------------------------------------------------------------------
    // RUN UI
    // ---------------------------------------------------------------------

    private void BeginRunUi(int modelCount)
    {
        _modelCount = modelCount;

        StartLatencyTestButton.IsEnabled = false;
        LatencyProgressBar.Value = 0;

        LatencyAverageText.Text = "- ms";
        LatencyMinText.Text = "- ms";
        LatencyMaxText.Text = "- ms";

        EndToEndAverageText.Text = "- ms";
        EndToEndMinText.Text = "- ms";
        EndToEndMaxText.Text = "- ms";

        ExecutionSummaryText.Text = "-";
        TimingMethodSummaryText.Text =
            "Timing method: -";

        _frontalLatencyMeasurements.Clear();
        _lateralLatencyMeasurements.Clear();

        FrontalLatencyGrid.Items.Refresh();
        LateralLatencyGrid.Items.Refresh();

        ResetViewLatencySummaries();

        FrontalLatencyStatusText.Text =
            "Collecting frontal per-case / " +
            "per-fold GPU-safe backend timings...";

        LateralLatencyStatusText.Text =
            "Collecting lateral per-case / " +
            "per-fold GPU-safe backend timings...";

        foreach (var latencyCase in _latencyCases)
        {
            latencyCase.ResetForRun();
        }

        LatencyResultsGrid.Items.Refresh();
    }

    private void OnFoldCompleted(
        ViewLatencyMeasurement frontal,
        ViewLatencyMeasurement lateral)
    {
        _frontalLatencyMeasurements.Add(frontal);
        _lateralLatencyMeasurements.Add(lateral);

        FrontalLatencyGrid.Items.Refresh();
        LateralLatencyGrid.Items.Refresh();

        UpdateViewLatencySummaries();
    }

    private static void ApplySuccessfulCaseResult(
        LatencyCase latencyCase,
        LatencyCaseRunResult result)
    {
        latencyCase.LatencyValueMilliseconds =
            result.EndToEndMilliseconds;

        latencyCase.LatencyMilliseconds =
            $"{result.EndToEndMilliseconds:F2}";

        latencyCase.BackendInferenceValueMilliseconds =
            result.BackendInferenceMilliseconds;

        latencyCase.BackendInferenceMilliseconds =
            $"{result.BackendInferenceMilliseconds:F2}";

        latencyCase.Classification =
            result.HasThrombus
                ? "Thrombus detected"
                : "No Thrombus detected";

        latencyCase.ExecutionProvider =
            result.ExecutionProvider;
        latencyCase.TimingDevice =
            result.TimingDevice;
        latencyCase.TimingMethod =
            result.TimingMethod;
        latencyCase.Status = "Complete";
    }

    private static void ApplyFailedCaseResult(
        LatencyCase latencyCase)
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
    }

    private void CompleteRunUi()
    {
        UpdateCaseLatencySummaries();
        UpdateViewLatencySummaries();
        UpdateViewStatusTexts();

        var failedCases =
            _latencyCases.Count(
                c => c.Status == "Failed");

        LatencyProgressText.Text =
            failedCases == 0
                ? $"Completed {_latencyCases.Count} of " +
                  $"{_latencyCases.Count} cases."
                : $"Completed " +
                  $"{_latencyCases.Count - failedCases} " +
                  $"cases. {failedCases} failed.";

        StartLatencyTestButton.IsEnabled = true;
        StartLatencyTestButton.ToolTip =
            "Run latency test again";
    }

    // ---------------------------------------------------------------------
    // SUMMARY RENDERING
    // ---------------------------------------------------------------------

    private void UpdateCaseLatencySummaries()
    {
        var inferenceSummary =
            LatencyStatistics.Calculate(
                _latencyCases
                    .Where(
                        c => c
                            .BackendInferenceValueMilliseconds
                            .HasValue)
                    .Select(
                        c => c
                            .BackendInferenceValueMilliseconds!
                            .Value));

        SetSummaryText(
            inferenceSummary,
            LatencyAverageText,
            LatencyMinText,
            LatencyMaxText);

        var endToEndSummary =
            LatencyStatistics.Calculate(
                _latencyCases
                    .Where(
                        c => c
                            .LatencyValueMilliseconds
                            .HasValue)
                    .Select(
                        c => c
                            .LatencyValueMilliseconds!
                            .Value));

        SetSummaryText(
            endToEndSummary,
            EndToEndAverageText,
            EndToEndMinText,
            EndToEndMaxText);

        var executionDescriptions =
            _latencyCases
                .Where(
                    c => c
                        .BackendInferenceValueMilliseconds
                        .HasValue)
                .Select(
                    c =>
                        $"{c.ExecutionProvider} " +
                        $"({c.TimingDevice})")
                .Distinct(
                    StringComparer.OrdinalIgnoreCase)
                .ToList();

        ExecutionSummaryText.Text =
            executionDescriptions.Count > 0
                ? string.Join(
                    "; ",
                    executionDescriptions)
                : "-";

        var timingMethods =
            _latencyCases
                .Where(
                    c => c
                        .BackendInferenceValueMilliseconds
                        .HasValue)
                .Select(c => c.TimingMethod)
                .Where(
                    method =>
                        !string.IsNullOrWhiteSpace(method) &&
                        method != "-")
                .Distinct(
                    StringComparer.OrdinalIgnoreCase)
                .ToList();

        TimingMethodSummaryText.Text =
            timingMethods.Count > 0
                ? $"Timing method: " +
                  $"{string.Join("; ", timingMethods)}"
                : "Timing method: -";
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

        TotalSelectedSequenceText.Text =
            (_latencyCases.Count * 2).ToString();

        TotalModelCountText.Text =
            _modelCount > 0
                ? _modelCount.ToString()
                : "-";

        TotalViewMeasurementText.Text =
            (_frontalLatencyMeasurements.Count +
             _lateralLatencyMeasurements.Count)
            .ToString();
    }

    private static void UpdateSingleViewSummary(
        IReadOnlyCollection<ViewLatencyMeasurement>
            measurements,
        System.Windows.Controls.TextBlock
            sequenceCountText,
        System.Windows.Controls.TextBlock
            measurementCountText,
        System.Windows.Controls.TextBlock minText,
        System.Windows.Controls.TextBlock averageText,
        System.Windows.Controls.TextBlock maxText)
    {
        sequenceCountText.Text =
            measurements
                .Select(m => m.CaseName)
                .Distinct(
                    StringComparer.OrdinalIgnoreCase)
                .Count()
                .ToString();

        measurementCountText.Text =
            measurements.Count.ToString();

        var summary =
            LatencyStatistics.Calculate(
                measurements.Select(
                    m => m.LatencyMilliseconds));

        SetSummaryText(
            summary,
            averageText,
            minText,
            maxText);
    }

    private static void SetSummaryText(
        LatencyMetricSummary summary,
        System.Windows.Controls.TextBlock averageText,
        System.Windows.Controls.TextBlock minText,
        System.Windows.Controls.TextBlock maxText)
    {
        averageText.Text =
            summary.Mean.HasValue
                ? $"{summary.Mean.Value:F2} ms"
                : "- ms";

        minText.Text =
            summary.Min.HasValue
                ? $"{summary.Min.Value:F2} ms"
                : "- ms";

        maxText.Text =
            summary.Max.HasValue
                ? $"{summary.Max.Value:F2} ms"
                : "- ms";
    }

    private void UpdateViewStatusTexts()
    {
        FrontalLatencyStatusText.Text =
            BuildViewStatusText(
                "Frontal",
                _frontalLatencyMeasurements);

        LateralLatencyStatusText.Text =
            BuildViewStatusText(
                "Lateral",
                _lateralLatencyMeasurements);
    }

    private static string BuildViewStatusText(
        string viewName,
        IReadOnlyCollection<ViewLatencyMeasurement>
            measurements)
    {
        if (measurements.Count == 0)
        {
            return
                $"No successful " +
                $"{viewName.ToLowerInvariant()} " +
                $"fold measurements were collected.";
        }

        var executionDescription =
            string.Join(
                "; ",
                measurements
                    .Select(
                        m =>
                            $"{m.ExecutionProvider} " +
                            $"({m.TimingDevice}) - " +
                            $"{m.TimingMethod}")
                    .Distinct());

        var sequenceCount =
            measurements
                .Select(m => m.CaseName)
                .Distinct(
                    StringComparer.OrdinalIgnoreCase)
                .Count();

        var foldCount =
            measurements
                .Select(m => m.ModelName)
                .Distinct(
                    StringComparer.OrdinalIgnoreCase)
                .Count();

        return
            $"{viewName}: {sequenceCount} sequence(s), " +
            $"{foldCount} fold(s), " +
            $"{measurements.Count} successful fold " +
            $"inference measurement(s). " +
            $"Execution: {executionDescription}.";
    }

    private void ResetAfterDatasetScan()
    {
        _frontalLatencyMeasurements.Clear();
        _lateralLatencyMeasurements.Clear();
        _modelCount = 0;

        LatencyResultsGrid.Items.Refresh();
        FrontalLatencyGrid.Items.Refresh();
        LateralLatencyGrid.Items.Refresh();

        LatencyCaseCountText.Text =
            _latencyCases.Count.ToString();

        TotalSelectedSequenceText.Text =
            (_latencyCases.Count * 2).ToString();

        LatencyAverageText.Text = "- ms";
        LatencyMinText.Text = "- ms";
        LatencyMaxText.Text = "- ms";

        EndToEndAverageText.Text = "- ms";
        EndToEndMinText.Text = "- ms";
        EndToEndMaxText.Text = "- ms";

        ExecutionSummaryText.Text = "-";
        TimingMethodSummaryText.Text =
            "Timing method: -";

        LatencyProgressBar.Value = 0;

        ResetViewLatencySummaries();

        if (_latencyCases.Count > 0)
        {
            LatencyProgressText.Text =
                $"{_latencyCases.Count} paired cases " +
                $"found and ready " +
                $"({_latencyCases.Count} frontal + " +
                $"{_latencyCases.Count} lateral sequences).";

            StartLatencyTestButton.IsEnabled = true;
            StartLatencyTestButton.ToolTip =
                "Start latency test";
        }
        else
        {
            LatencyProgressText.Text =
                "No valid frontal/lateral pairs " +
                "were found.";

            StartLatencyTestButton.IsEnabled = false;
            StartLatencyTestButton.ToolTip =
                "No valid dataset pairs were found.";
        }
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

        TotalModelCountText.Text =
            _modelCount > 0
                ? _modelCount.ToString()
                : "-";

        TotalViewMeasurementText.Text = "0";

        FrontalLatencyStatusText.Text =
            "Run the test to collect frontal " +
            "per-case / per-fold timings.";

        LateralLatencyStatusText.Text =
            "Run the test to collect lateral " +
            "per-case / per-fold timings.";
    }
}
