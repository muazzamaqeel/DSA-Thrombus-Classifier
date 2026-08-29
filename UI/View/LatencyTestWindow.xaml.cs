using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Threading;
using MaterialDesignThemes.Wpf;
using Microsoft.WindowsAPICodePack.Dialogs;
using UI.LatencyTest;

namespace UI.View;

public partial class LatencyTestWindow : Window
{
    private static readonly string[] PaperFoldModelNames =
        ["fold1.pt", "fold2.pt", "fold3.pt", "fold4.pt", "fold5.pt"];

    private readonly LatencyDatasetScanner _scanner = new();
    private readonly LatencyTestRunner _runner = new();
    private readonly ObservableCollection<LatencyCase> _cases = new();
    private readonly ObservableCollection<ViewLatencyMeasurement> _frontal = new();
    private readonly ObservableCollection<ViewLatencyMeasurement> _lateral = new();

    public static ICommand OpenCommand { get; } =
        new UI.RelayCommand<Window>(owner =>
        {
            new LatencyTestWindow
            {
                Owner = owner,
                DataContext = owner.DataContext
            }.ShowDialog();
        });

    public LatencyTestWindow()
    {
        InitializeComponent();
        LatencyResultsGrid.ItemsSource = _cases;
        FrontalLatencyGrid.ItemsSource = _frontal;
        LateralLatencyGrid.ItemsSource = _lateral;
        ResetSummaries();
    }

    private async void SelectLatencyDataset_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new CommonOpenFileDialog
        {
            IsFolderPicker = true,
            Title = "Select latency test dataset folder"
        };

        if (dialog.ShowDialog() != CommonFileDialogResult.Ok)
            return;

        LatencyDatasetPathText.Text = dialog.FileName;
        LatencyProgressText.Text = "Scanning dataset...";
        StartLatencyTestButton.IsEnabled = false;

        var found = await Task.Run(() => _scanner.Scan(dialog.FileName));
        _cases.Clear();
        foreach (var item in found)
            _cases.Add(item);

        _frontal.Clear();
        _lateral.Clear();
        ResetSummaries();
        LatencyProgressBar.Value = 0;
        LatencyResultsGrid.Items.Refresh();

        StartLatencyTestButton.IsEnabled = _cases.Count > 0;
        StartLatencyTestButton.ToolTip = _cases.Count > 0
            ? "Start latency test"
            : "No valid dataset pairs were found.";
        LatencyProgressText.Text = _cases.Count > 0
            ? $"{_cases.Count} paired cases found and ready."
            : "No valid frontal/lateral pairs were found.";
    }

    private void StartLatencyTestSurface_MouseLeftButtonUp(
        object sender, MouseButtonEventArgs e) => StartLatencyTestButton_Click(sender, e);

    private async void StartLatencyTestButton_Click(object sender, RoutedEventArgs e)
    {
        if (!TryGetPaperModels(out var models))
            return;

        BeginRun();

        var requestedMode =
            (LatencyExecutionModeComboBox.SelectedItem as ComboBoxItem)?
                .Content?.ToString() ?? "GPU";

        LatencyProgressText.Text =
            $"Configuring {requestedMode} execution...";

        try
        {
            var vm = (MainWindowViewModel)DataContext;
            var execution = await _runner.ConfigureExecutionAsync(
                requestedMode, vm.ModelSelectionFolder);

            ExecutionSummaryText.Text =
                $"{execution.ExecutionProvider} ({execution.TimingDevice})";
        }
        catch (Exception error)
        {
            StartLatencyTestButton.IsEnabled = true;
            LatencyExecutionModeComboBox.IsEnabled = true;
            LatencyProgressText.Text = "Execution-device configuration failed.";

            MessageBox.Show(
                error.Message,
                "Latency Test - Execution Device",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return;
        }

        var failures = new List<string>();
        var completed = 0;

        foreach (var latencyCase in _cases)
        {
            latencyCase.Status = "Processing";
            LatencyProgressText.Text =
                $"Processing {completed + 1} of {_cases.Count}: {latencyCase.CaseName}";
            LatencyResultsGrid.Items.Refresh();

            try
            {
                var result = await _runner.RunCaseAsync(
                    latencyCase, models, LatencyThresholdSlider.Value);

                ApplyResult(latencyCase, result);
                foreach (var row in result.FrontalMeasurements) _frontal.Add(row);
                foreach (var row in result.LateralMeasurements) _lateral.Add(row);
            }
            catch (Exception error)
            {
                latencyCase.Status = "Failed";
                latencyCase.Classification = "Error";
                failures.Add($"{latencyCase.CaseName}: {error.Message}");
            }

            completed++;
            LatencyProgressBar.Value = (double)completed / _cases.Count * 100.0;
            LatencyResultsGrid.Items.Refresh();
            UpdateSummaries();
            await Dispatcher.Yield(DispatcherPriority.Background);
        }

        StartLatencyTestButton.IsEnabled = true;
        LatencyExecutionModeComboBox.IsEnabled = true;
        StartLatencyTestButton.ToolTip = "Run latency test again";
        LatencyProgressText.Text = failures.Count == 0
            ? $"Completed {_cases.Count} of {_cases.Count} cases."
            : $"Completed {_cases.Count - failures.Count} cases. {failures.Count} failed.";

        if (failures.Count > 0)
        {
            var shown = string.Join("\n", failures.Take(10));
            var more = failures.Count > 10 ? $"\n... and {failures.Count - 10} more." : "";
            MessageBox.Show(
                $"{failures.Count} case(s) failed.\n\n{shown}{more}",
                "Latency Test", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private bool TryGetPaperModels(out IReadOnlyCollection<string> models)
    {
        models = PaperFoldModelNames;

        if (_cases.Count == 0)
            return Warn("No dataset cases are available.");

        if (DataContext is not MainWindowViewModel vm)
            return Warn("The classifier view model is unavailable.");

        if (string.IsNullOrWhiteSpace(vm.ModelSelectionFolder))
            return Warn("Please select a model folder first.");

        if (vm.ModelSelectionFolderBadge.Kind != PackIconKind.Check)
            return Warn("Please wait until model initialization has completed.");

        var frontal = Path.Combine(vm.ModelSelectionFolder, "frontal");
        var lateral = Path.Combine(vm.ModelSelectionFolder, "lateral");
        if (!Directory.Exists(frontal) || !Directory.Exists(lateral))
            return Warn("The model folder must contain frontal and lateral directories.");

        var expected = PaperFoldModelNames.OrderBy(x => x, StringComparer.Ordinal);
        if (!ModelNames(frontal).SequenceEqual(expected, StringComparer.Ordinal) ||
            !ModelNames(lateral).SequenceEqual(expected, StringComparer.Ordinal))
        {
            return Warn(
                "For paper reproduction, frontal and lateral must each contain " +
                "exactly fold1.pt through fold5.pt.");
        }

        return true;
    }

    private static IEnumerable<string> ModelNames(string folder) =>
        Directory.GetFiles(folder, "*.pt")
            .Select(Path.GetFileName)
            .Where(name => name is not null)
            .Select(name => name!)
            .OrderBy(name => name, StringComparer.Ordinal);

    private static bool Warn(string message)
    {
        MessageBox.Show(message, "Latency Test", MessageBoxButton.OK, MessageBoxImage.Warning);
        return false;
    }

    private void BeginRun()
    {
        StartLatencyTestButton.IsEnabled = false;
        LatencyExecutionModeComboBox.IsEnabled = false;
        LatencyProgressBar.Value = 0;
        _frontal.Clear();
        _lateral.Clear();
        foreach (var item in _cases) item.Reset();
        LatencyResultsGrid.Items.Refresh();
        ResetSummaries();
    }

    private static void ApplyResult(LatencyCase latencyCase, LatencyCaseRunResult result)
    {
        latencyCase.Classification = result.HasThrombus
            ? "Thrombus detected"
            : "No Thrombus detected";
        latencyCase.InferenceMilliseconds = result.InferenceMilliseconds;
        latencyCase.ExecutionProvider = result.ExecutionProvider;
        latencyCase.TimingDevice = result.TimingDevice;
        latencyCase.Status = "Complete";
    }

    private void UpdateSummaries()
    {
        SetSummary(
            LatencyStatistics.Calculate(_cases
                .Where(x => x.InferenceMilliseconds.HasValue)
                .Select(x => x.InferenceMilliseconds!.Value)),
            LatencyAverageText, LatencyMinText, LatencyMaxText);

        UpdateViewSummary(
            _frontal, FrontalSequenceCountText, FrontalMeasurementCountText,
            FrontalMinText, FrontalAverageText, FrontalMaxText);
        UpdateViewSummary(
            _lateral, LateralSequenceCountText, LateralMeasurementCountText,
            LateralMinText, LateralAverageText, LateralMaxText);

        ExecutionSummaryText.Text = string.Join(
            "; ",
            _cases.Where(x => x.InferenceMilliseconds.HasValue)
                .Select(x => $"{x.ExecutionProvider} ({x.TimingDevice})")
                .Distinct(StringComparer.OrdinalIgnoreCase));

        if (string.IsNullOrWhiteSpace(ExecutionSummaryText.Text))
            ExecutionSummaryText.Text = "-";
    }

    private static void UpdateViewSummary(
        IReadOnlyCollection<ViewLatencyMeasurement> values,
        System.Windows.Controls.TextBlock sequenceCount,
        System.Windows.Controls.TextBlock measurementCount,
        System.Windows.Controls.TextBlock min,
        System.Windows.Controls.TextBlock mean,
        System.Windows.Controls.TextBlock max)
    {
        sequenceCount.Text = values.Select(x => x.CaseName)
            .Distinct(StringComparer.OrdinalIgnoreCase).Count().ToString();
        measurementCount.Text = values.Count.ToString();
        SetSummary(
            LatencyStatistics.Calculate(values.Select(x => x.LatencyMilliseconds)),
            mean, min, max);
    }

    private static void SetSummary(
        LatencyMetricSummary summary,
        System.Windows.Controls.TextBlock mean,
        System.Windows.Controls.TextBlock min,
        System.Windows.Controls.TextBlock max)
    {
        mean.Text = summary.Mean is double a ? $"{a:F2} ms" : "- ms";
        min.Text = summary.Min is double b ? $"{b:F2} ms" : "- ms";
        max.Text = summary.Max is double c ? $"{c:F2} ms" : "- ms";
    }

    private void ResetSummaries()
    {
        LatencyAverageText.Text = LatencyMinText.Text = LatencyMaxText.Text = "- ms";
        ExecutionSummaryText.Text = "-";
        FrontalSequenceCountText.Text = FrontalMeasurementCountText.Text = "0";
        LateralSequenceCountText.Text = LateralMeasurementCountText.Text = "0";
        FrontalMinText.Text = FrontalAverageText.Text = FrontalMaxText.Text = "- ms";
        LateralMinText.Text = LateralAverageText.Text = LateralMaxText.Text = "- ms";
    }
}
