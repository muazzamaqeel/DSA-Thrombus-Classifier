
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Threading;
using System.ComponentModel;
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
    private string? _latencyModelFolder;
    private bool _isRunning;
    private CancellationTokenSource? _runCancellation;
    private bool _isScanning;
    private bool _devicesLoaded;

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
        InitializeResultExportControls();
        Title = "Latency Test - v7 GPU batched single pass";
        Closing += OnLatencyWindowClosing;
        LatencyProgressBar.IsIndeterminate = false;
        LatencyProgressBar.Minimum = 0;
        LatencyProgressBar.Maximum = 100;
        LatencyExecutionModeComboBox.IsEnabled = false;
        Loaded += LoadBackendDevices;
        LatencyResultsGrid.ItemsSource = _cases;
        FrontalLatencyGrid.ItemsSource = _frontal;
        LateralLatencyGrid.ItemsSource = _lateral;
        ResetSummaries();
    }

    private async void LoadBackendDevices(object sender, RoutedEventArgs e)
    {
        Loaded -= LoadBackendDevices;
        try
        {
            var info = await _runner.GetInfoAsync();
            LatencyExecutionModeComboBox.Items.Clear();
            foreach (var device in info.Devices)
                LatencyExecutionModeComboBox.Items.Add(new ComboBoxItem
                {
                    Content = $"GPU {device.Index}: {device.Name}", Tag = device.Index
                });
            LatencyExecutionModeComboBox.Items.Add(new ComboBoxItem { Content = "CPU", Tag = -1 });
            LatencyExecutionModeComboBox.SelectedIndex = 0;
            LatencyExecutionModeComboBox.IsEnabled = true;
            LatencyExecutionModeComboBox.ToolTip = "Select the actual backend device used for inference.";
            ExecutionSummaryText.TextWrapping = TextWrapping.Wrap;
            _devicesLoaded = true;
            StartLatencyTestButton.IsEnabled = _cases.Count > 0 && !_isScanning;
            if (info.Devices.Count == 0)
                LatencyProgressText.Text = "CUDA is unavailable in this backend. CPU is selected.";
        }
        catch (Exception error)
        {
            StartLatencyTestButton.IsEnabled = false;
            LatencyProgressText.Text = "Update/restart the backend before starting the test.";
            MessageBox.Show(this, error.Message, "Latency Test - Backend", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private async void SelectLatencyDataset_Click(object sender, RoutedEventArgs e)
    {
        if (_isRunning || _isScanning)
            return;

        var dialog = new CommonOpenFileDialog
        {
            IsFolderPicker = true,
            Title = "Select latency test dataset folder"
        };

        if (dialog.ShowDialog() != CommonFileDialogResult.Ok)
            return;

        _isScanning = true;
        var exportWasEnabled = _csvExportButton.IsEnabled;
        _csvExportButton.IsEnabled = false;
        LatencyProgressText.Text = "Scanning dataset...";
        StartLatencyTestButton.IsEnabled = false;

        IReadOnlyList<LatencyCase> found;
        try
        {
            found = await Task.Run(() => _scanner.Scan(dialog.FileName));
        }
        catch (Exception error)
        {
            _csvExportButton.IsEnabled = exportWasEnabled;
            StartLatencyTestButton.IsEnabled = _cases.Count > 0 && _devicesLoaded;
            LatencyProgressText.Text = "Dataset scan failed.";
            MessageBox.Show(this, error.Message, "Latency Test",
                MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }
        finally
        {
            _isScanning = false;
        }

        LatencyDatasetPathText.Text = dialog.FileName;
        _cases.Clear();
        foreach (var item in found)
            _cases.Add(item);

        _frontal.Clear();
        _lateral.Clear();
        ResetSummaries();
        LatencyProgressBar.Value = 0;
        LatencyResultsGrid.Items.Refresh();

        StartLatencyTestButton.IsEnabled = _cases.Count > 0 && _devicesLoaded;
        StartLatencyTestButton.ToolTip = _cases.Count > 0
            ? "Start latency test"
            : "No valid dataset pairs were found.";
        LatencyProgressText.Text = _cases.Count > 0
            ? $"{_cases.Count} paired cases found and ready."
            : "No valid frontal/lateral pairs were found.";
    }

    private void StartLatencyTestSurface_MouseLeftButtonUp(
        object sender, MouseButtonEventArgs e) => StartLatencyTestButton_Click(sender, e);

    private async void StopLatencyTest_Click(object sender, RoutedEventArgs e)
    {
        if (!_isRunning) return;
        _runCancellation?.Cancel();
        _stopLatencyButton.IsEnabled = false;
        LatencyProgressText.Text = "Stopping after the active operation...";
        try { await _runner.CancelAsync(); }
        catch (Exception error) { LatencyProgressText.Text = $"Stop request failed: {error.Message}"; }
    }

    private void OnLatencyWindowClosing(object? sender, CancelEventArgs e)
    {
        if (!_isRunning) return;
        e.Cancel = true;
        StopLatencyTest_Click(this, new RoutedEventArgs());
    }

    private async void StartLatencyTestButton_Click(object sender, RoutedEventArgs e)
    {
        if (_isRunning || _isScanning || !TryGetPaperModels(out var models)) return;
        BeginRun();
        _runCancellation = new CancellationTokenSource();
        var token = _runCancellation.Token;
        var selected = LatencyExecutionModeComboBox.SelectedItem as ComboBoxItem;
        var deviceIndex = selected?.Tag is int index ? index : -1;
        var requestedMode = deviceIndex >= 0 ? "GPU" : "CPU";
        var threshold = LatencyThresholdSlider.Value;
        var completed = 0;
        LatencyCase? activeCase = null;
        try
        {
            LatencyProgressText.Text = $"Checking {requestedMode} and checkpoint fingerprints...";
            var execution = await _runner.ConfigureExecutionAsync(
                requestedMode, _latencyModelFolder!, Math.Max(deviceIndex, 0));
            token.ThrowIfCancellationRequested();
            ExecutionSummaryText.Text = $"{execution.ExecutionProvider} ({execution.TimingDevice})";
            foreach (var latencyCase in _cases)
            {
                token.ThrowIfCancellationRequested();
                activeCase = latencyCase;
                latencyCase.Status = "Processing";
                LatencyProgressText.Text = $"Processing {completed + 1} of {_cases.Count}: {latencyCase.CaseName}";
                LatencyResultsGrid.Items.Refresh();
                var result = await _runner.RunCaseAsync(latencyCase, models, threshold, token);
                token.ThrowIfCancellationRequested();
                ApplyResult(latencyCase, result);
                foreach (var row in result.FrontalMeasurements) _frontal.Add(row);
                foreach (var row in result.LateralMeasurements) _lateral.Add(row);
                completed++;
                activeCase = null;
                LatencyProgressBar.Value = (double)completed / _cases.Count * 100;
                LatencyResultsGrid.Items.Refresh();
                UpdateSummaries();
                await Dispatcher.Yield(DispatcherPriority.Background);
            }
            LatencyProgressText.Text = $"Completed {completed} of {_cases.Count} cases. One measured pass per view/fold.";
        }
        catch (Exception error)
        {
            var cancelled = token.IsCancellationRequested;
            if (activeCase is not null)
            {
                activeCase.Status = cancelled ? "Cancelled" : "Failed";
                activeCase.Classification = cancelled ? "-" : "Error";
            }
            foreach (var item in _cases.Where(x => x.Status == "Ready")) item.Status = "Not run";
            LatencyProgressText.Text = cancelled
                ? $"Stopped. {completed} completed cases are available for CSV export."
                : $"Test stopped after {completed} cases: {error.Message}";
            try { await _runner.CancelAsync(); } catch (Exception) { /* Keep the original failure. */ }
            if (!cancelled)
                MessageBox.Show(this, error.Message, "Latency test stopped", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        finally
        {
            try { await _runner.EndAsync(); }
            catch (Exception error)
            {
                LatencyProgressText.Text += $" Cleanup pending: {error.Message}. Restart the backend if it remains busy.";
            }
            _isRunning = false;
            _runCancellation.Dispose();
            _runCancellation = null;
            _stopLatencyButton.IsEnabled = false;
            _latencyModelsButton.IsEnabled = true;
            _csvExportButton.IsEnabled = _cases.Count > 0;
            StartLatencyTestButton.IsEnabled = _devicesLoaded && _cases.Count > 0;
            LatencyExecutionModeComboBox.IsEnabled = true;
            LatencyThresholdSlider.IsEnabled = true;
            LatencyResultsGrid.Items.Refresh();
            UpdateSummaries();
        }
    }

    private bool TryGetPaperModels(out IReadOnlyCollection<string> models)
    {
        models = PaperFoldModelNames;
        if (!_devicesLoaded)
            return Warn("Backend GPU information is not ready. Verify that the updated backend is running.");

        if (_cases.Count == 0)
            return Warn("No dataset cases are available.");

        if (string.IsNullOrWhiteSpace(_latencyModelFolder) && !SelectLatencyModels())
            return false;
        var frontal = Path.Combine(_latencyModelFolder!, "frontal");
        var lateral = Path.Combine(_latencyModelFolder!, "lateral");
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

    private bool SelectLatencyModels()
    {
        if (_isRunning) return false;
        using var dialog = new CommonOpenFileDialog
        {
            IsFolderPicker = true, Title = "Select latency models (frontal and lateral folders)"
        };
        if (dialog.ShowDialog() != CommonFileDialogResult.Ok) return false;
        _latencyModelFolder = dialog.FileName;
        _latencyModelsButton.ToolTip = _latencyModelFolder;
        _latencyModelsButton.Content = "Change latency models";
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
        _isRunning = true;
        _latencyModelsButton.IsEnabled = false;
        _stopLatencyButton.IsEnabled = true;
        LatencyThresholdSlider.IsEnabled = false;
        _csvExportButton.IsEnabled = false;
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
        latencyCase.TimingMethod = result.TimingMethod;
        latencyCase.ModelResidency = result.ModelResidency;
        latencyCase.EnvironmentJson = result.EnvironmentJson;
        latencyCase.PreparationJson = result.PreparationJson;
        latencyCase.ClientCaseMilliseconds = result.ClientCaseMilliseconds;
        latencyCase.ModelOutput = result.ModelOutput;
        latencyCase.Threshold = result.Threshold;
        latencyCase.Classification = result.HasThrombus
            ? "Thrombus detected"
            : "No Thrombus detected";
        latencyCase.InferenceMilliseconds = result.InferenceMilliseconds;
        latencyCase.ForwardMilliseconds = result.ForwardMilliseconds;
        latencyCase.ExecutionProvider = result.ExecutionProvider;
        latencyCase.TimingDevice = result.TimingDevice;
        latencyCase.Status = "Complete";
    }

    private void UpdateSummaries()
    {
        var summary = LatencyStatistics.Calculate(_cases.Where(x => x.Status == "Complete" && x.InferenceMilliseconds.HasValue)
            .Select(x => x.InferenceMilliseconds!.Value));
        _classificationStdText.Text = summary.StandardDeviation is double sd
            ? $"Sample SD: {sd:F2} ms (n={summary.Count})" : $"Sample SD: - (n={summary.Count})";
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
        _classificationStdText.Text = "Sample SD: - (n=0)";
        ExecutionSummaryText.Text = "-";
        FrontalSequenceCountText.Text = FrontalMeasurementCountText.Text = "0";
        LateralSequenceCountText.Text = LateralMeasurementCountText.Text = "0";
        FrontalMinText.Text = FrontalAverageText.Text = FrontalMaxText.Text = "- ms";
        LateralMinText.Text = LateralAverageText.Text = LateralMaxText.Text = "- ms";
    }
}
