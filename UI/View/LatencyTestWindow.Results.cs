using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using Microsoft.Win32;
using UI.LatencyTest;

namespace UI.View;

public partial class LatencyTestWindow
{
    private readonly Button _latencyModelsButton = new()
    {
        Content = "Select latency models", MinWidth = 120, MinHeight = 36,
        HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(0, 10, 0, 0)
    };
    private readonly Button _stopLatencyButton = new()
    {
        Content = "Stop test", IsEnabled = false, MinWidth = 120, MinHeight = 36,
        HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(0, 10, 0, 0)
    };
    private readonly TextBlock _classificationStdText = new()
    {
        Text = "Sample SD: - (n=0)", Margin = new Thickness(0, 10, 0, 0), TextWrapping = TextWrapping.Wrap
    };

    private readonly Button _csvExportButton = new()
    {
        Content = "CSV Export",
        IsEnabled = false,
        MinWidth = 120,
        MinHeight = 36,
        HorizontalAlignment = HorizontalAlignment.Right,
        Margin = new Thickness(0, 10, 0, 0),
        ToolTip = "Export the completed test results to a CSV file."
    };

    private readonly List<(FrameworkElement Element, Visibility Original)> _timingSections = new();
    private readonly List<(ColumnDefinition Column, GridLength Width, double Min)> _timingColumns = new();
    private readonly List<(RowDefinition Row, GridLength Height, double Min)> _timingRows = new();

    private void InitializeResultExportControls()
    {
        // Also support a grid that uses auto-generated columns in the existing XAML.
        // These new storage properties must not introduce unwanted extra columns.
        LatencyResultsGrid.AutoGeneratingColumn += (_, e) =>
        {
            if (e.PropertyName is nameof(LatencyCase.ModelOutput) or nameof(LatencyCase.Threshold)
                or nameof(LatencyCase.TimingMethod) or nameof(LatencyCase.ModelResidency)
                or nameof(LatencyCase.EnvironmentJson) or nameof(LatencyCase.PreparationJson)
                or nameof(LatencyCase.ClientCaseMilliseconds) or nameof(LatencyCase.ForwardMilliseconds))
                e.Cancel = true;
        };
        FrontalLatencyGrid.AutoGeneratingColumn += HideExtraMeasurementColumns;
        LateralLatencyGrid.AutoGeneratingColumn += HideExtraMeasurementColumns;

        AddModelOutputColumn(LatencyResultsGrid, nameof(LatencyCase.Classification), "Classification");
        AddModelOutputColumn(FrontalLatencyGrid, nameof(ViewLatencyMeasurement.ModelName), "Model / Fold");
        AddModelOutputColumn(LateralLatencyGrid, nameof(ViewLatencyMeasurement.ModelName), "Model / Fold");

        foreach (var grid in new[] { LatencyResultsGrid, FrontalLatencyGrid, LateralLatencyGrid })
        {
            ScrollViewer.SetIsDeferredScrollingEnabled(grid, false);
            ScrollViewer.SetCanContentScroll(grid, true);
            VirtualizingPanel.SetScrollUnit(grid, ScrollUnit.Pixel);
            grid.Loaded += (_, _) => ConfigureGridScrolling(grid);
            grid.PreviewMouseWheel += ResultGrid_PreviewMouseWheel;
        }
        Loaded += (_, _) =>
        {
            ConfigureEvaluationScrolling();
            ApplyForwardTimingLabels();
        };

        // Keep the user's original XAML, resources and all existing controls.
        // The new actions sit directly below the named EXECUTION value.
        var actions = new StackPanel { HorizontalAlignment = HorizontalAlignment.Right };
        _csvExportButton.Click += CsvExport_Click;
        _latencyModelsButton.Click += (_, _) => SelectLatencyModels();
        actions.Children.Add(_latencyModelsButton);
        actions.Children.Add(_classificationStdText);
        _stopLatencyButton.Click += StopLatencyTest_Click;
        actions.Children.Add(_stopLatencyButton);
        actions.Children.Add(_csvExportButton);

        var showTiming = new CheckBox
        {
            Content = "Show per-fold timing",
            IsChecked = false,
            Margin = new Thickness(0, 10, 0, 0),
            HorizontalAlignment = HorizontalAlignment.Right,
            ToolTip = "Show or hide the frontal and lateral per-fold timing sections."
        };
        showTiming.Checked += (_, _) =>
        {
            SetPerFoldTimingVisible(true);
            Dispatcher.BeginInvoke(DispatcherPriority.Loaded,
                new Action(ConfigureEvaluationScrolling));
        };
        showTiming.Unchecked += (_, _) => SetPerFoldTimingVisible(false);
        actions.Children.Add(showTiming);
        AddBelowExecutionSummary(actions);

        InitializeTimingSections();
        SetPerFoldTimingVisible(false);
    }

    private static void HideExtraMeasurementColumns(object? sender, DataGridAutoGeneratingColumnEventArgs e)
    {
        if (e.PropertyName is nameof(ViewLatencyMeasurement.ModelOutput) or nameof(ViewLatencyMeasurement.CaseKey)
            or nameof(ViewLatencyMeasurement.BenchmarkJson) or nameof(ViewLatencyMeasurement.ForwardMilliseconds))
            e.Cancel = true;
    }

    private static void AddModelOutputColumn(DataGrid grid, string precedingProperty, string precedingHeader)
    {
        // Both LatencyCase and ViewLatencyMeasurement expose the stored ModelOutput.
        var output = new DataGridTextColumn
        {
            Header = "Model output",
            Binding = new Binding(nameof(LatencyCase.ModelOutput))
            {
                Mode = BindingMode.OneWay,
                StringFormat = "{0:G17}",
                ConverterCulture = CultureInfo.InvariantCulture,
                TargetNullValue = "-"
            },
            SortMemberPath = nameof(LatencyCase.ModelOutput),
            IsReadOnly = true,
            MinWidth = 190,
            Width = DataGridLength.SizeToCells
        };
        grid.Columns.Add(output);

        void PositionOutput()
        {
            // Header matching also finds DataGridTemplateColumn definitions, which
            // do not expose DataGridBoundColumn.Binding. Preserve every other column's order.
            var ordered = grid.Columns.Where(x => x != output)
                .OrderBy(x => x.DisplayIndex < 0 ? grid.Columns.IndexOf(x) : x.DisplayIndex)
                .ToList();
            var preceding = ordered.FindIndex(x =>
                x.SortMemberPath == precedingProperty ||
                (x is DataGridBoundColumn bound &&
                 (bound.Binding as Binding)?.Path?.Path == precedingProperty) ||
                NormalizeHeader(x.Header) == NormalizeHeader(precedingHeader));
            if (preceding < 0)
                return; // Auto-generated columns may not exist until the grid is loaded.

            ordered.Insert(preceding + 1, output);
            for (var index = 0; index < ordered.Count; index++)
                ordered[index].DisplayIndex = index;
        }

        PositionOutput();
        grid.AutoGeneratedColumns += (_, _) => PositionOutput();
        grid.Loaded += (_, _) =>
        {
            PositionOutput();
            // Run once more after initial XAML/template bindings have settled.
            grid.Dispatcher.BeginInvoke(DispatcherPriority.Loaded, new Action(PositionOutput));
        };
    }

    private static string NormalizeHeader(object? header)
    {
        var text = header switch
        {
            TextBlock block => block.Text,
            AccessText access => access.Text,
            ContentControl content => content.Content?.ToString() ?? "",
            _ => header?.ToString() ?? ""
        };
        return new string(text.Where(char.IsLetterOrDigit).ToArray()).ToUpperInvariant();
    }

    private void ConfigureEvaluationScrolling()
    {
        ConfigureGridScrolling(LatencyResultsGrid);
        ConfigureGridScrolling(FrontalLatencyGrid);
        ConfigureGridScrolling(LateralLatencyGrid);
    }

    private static void ConfigureGridScrolling(DataGrid grid)
    {
        // Keep row virtualization, while allowing positions between row boundaries.
        ScrollViewer.SetCanContentScroll(grid, true);
        VirtualizingPanel.SetScrollUnit(grid, ScrollUnit.Pixel);
        ScrollViewer.SetIsDeferredScrollingEnabled(grid, false);
        grid.ApplyTemplate();
        var tableViewer = FindGridScrollViewer(grid);
        if (tableViewer is not null)
        {
            tableViewer.CanContentScroll = true;
            tableViewer.IsDeferredScrollingEnabled = false;
        }

        // A page StackPanel can otherwise scroll by whole cards. Use physical
        // scrolling on the enclosing page, and update its content during dragging.
        foreach (var pageViewer in VisualAncestors(grid).OfType<ScrollViewer>())
        {
            pageViewer.CanContentScroll = false;
            pageViewer.IsDeferredScrollingEnabled = false;
        }
    }

    private static void ResultGrid_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
    {
        if (e.Handled || e.Delta == 0 || Keyboard.Modifiers != ModifierKeys.None || sender is not DataGrid grid)
            return;

        var viewer = FindGridScrollViewer(grid);
        if (viewer is null)
            return;

        var atBoundary = e.Delta > 0
            ? viewer.VerticalOffset <= 0.5
            : viewer.VerticalOffset >= viewer.ScrollableHeight - 0.5;
        if (!atBoundary)
            return; // The table handles ordinary wheel scrolling itself.

        var pageViewer = VisualAncestors(grid).OfType<ScrollViewer>().FirstOrDefault();
        if (pageViewer is null)
            return;

        // Preserve Windows' wheel settings and delta. Only transfer the wheel
        // to the containing page when the table cannot scroll farther.
        e.Handled = true;
        pageViewer.RaiseEvent(new MouseWheelEventArgs(e.MouseDevice, e.Timestamp, e.Delta)
        {
            RoutedEvent = Mouse.MouseWheelEvent,
            Source = pageViewer
        });
    }

    private static ScrollViewer? FindGridScrollViewer(DataGrid grid) =>
        grid.Template?.FindName("DG_ScrollViewer", grid) as ScrollViewer ??
        VisualDescendants<ScrollViewer>(grid).FirstOrDefault();

    private static IEnumerable<DependencyObject> VisualAncestors(DependencyObject element)
    {
        for (var parent = VisualTreeHelper.GetParent(element); parent is not null;
             parent = VisualTreeHelper.GetParent(parent))
            yield return parent;
    }

    private static IEnumerable<T> VisualDescendants<T>(DependencyObject element) where T : DependencyObject
    {
        for (var index = 0; index < VisualTreeHelper.GetChildrenCount(element); index++)
        {
            var child = VisualTreeHelper.GetChild(element, index);
            if (child is T match)
                yield return match;
            foreach (var descendant in VisualDescendants<T>(child))
                yield return descendant;
        }
    }

    private void ApplyForwardTimingLabels()
    {
        foreach (var grid in new[] { LatencyResultsGrid, FrontalLatencyGrid, LateralLatencyGrid })
        {
            var name = grid == LatencyResultsGrid ? "Ensemble" : grid == FrontalLatencyGrid ? "Frontal" : "Lateral";
            foreach (var column in grid.Columns)
            {
                if ((NormalizeHeader(column.Header).Contains("INFERENCE") || NormalizeHeader(column.Header).Contains("FORWARD")))
                    column.Header = $"{name} classification (ms)";
            }
        }
        foreach (var text in VisualDescendants<TextBlock>(this))
        {
            if (text.Text.Equals("MEAN INFERENCE", StringComparison.OrdinalIgnoreCase))
                text.Text = "MEAN CLASSIFICATION";
            if (text.Text.StartsWith("Each row represents a paired DSA case.", StringComparison.Ordinal))
                text.Text = "Each row is a paired case: one measured prediction per view and fold, plus soft voting. Loading, preprocessing, transfers and the first-case warm-up are excluded.";
        }
        LatencyResultsGrid.ToolTip = "Classification uses synchronized wall time with one pass per view/fold. CSV also records forward time, transfers, warm-up and client processing time. Mean and sample SD are across completed cases.";
    }

    private void CsvExport_Click(object sender, RoutedEventArgs e)
    {
        if (_isRunning || _isScanning || !_csvExportButton.IsEnabled || _cases.Count == 0)
            return;

        var dialog = new SaveFileDialog
        {
            Title = "Export latency test results",
            Filter = "CSV files (*.csv)|*.csv",
            DefaultExt = ".csv",
            AddExtension = true,
            OverwritePrompt = true,
            CheckPathExists = true,
            FileName = $"Latency_Test_Results_{DateTime.Now:yyyyMMdd_HHmmss}.csv"
        };

        if (dialog.ShowDialog(this) != true)
            return;

        try
        {
            // Export every dataset row, independent of grid sorting or hidden timing sections.
            LatencyCsvExporter.Write(dialog.FileName, _cases, _frontal, _lateral,
                PaperFoldModelNames);
            MessageBox.Show(this, $"CSV exported to:\n{dialog.FileName}",
                "CSV Export", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (Exception error)
        {
            MessageBox.Show(this, $"Could not export the CSV file.\n\n{error.Message}",
                "CSV Export", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void AddBelowExecutionSummary(FrameworkElement actions)
    {
        var value = ExecutionSummaryText;
        var parent = LogicalTreeHelper.GetParent(value);
        if (parent is StackPanel { Orientation: Orientation.Vertical } stack)
        {
            stack.Children.Insert(stack.Children.IndexOf(value) + 1, actions);
            return;
        }

        // Grid/content layouts need a vertical wrapper in the value's existing slot.
        var wrapper = new StackPanel();
        Grid.SetRow(wrapper, Grid.GetRow(value));
        Grid.SetColumn(wrapper, Grid.GetColumn(value));
        Grid.SetRowSpan(wrapper, Grid.GetRowSpan(value));
        Grid.SetColumnSpan(wrapper, Grid.GetColumnSpan(value));
        DockPanel.SetDock(wrapper, DockPanel.GetDock(value));
        Panel.SetZIndex(wrapper, Panel.GetZIndex(value));
        Canvas.SetLeft(wrapper, Canvas.GetLeft(value));
        Canvas.SetTop(wrapper, Canvas.GetTop(value));
        Canvas.SetRight(wrapper, Canvas.GetRight(value));
        Canvas.SetBottom(wrapper, Canvas.GetBottom(value));

        switch (parent)
        {
            case Panel panel:
                var index = panel.Children.IndexOf(value);
                panel.Children.RemoveAt(index);
                panel.Children.Insert(index, wrapper);
                if (panel is Grid grid && Grid.GetRowSpan(value) == 1 &&
                    Grid.GetRow(value) < grid.RowDefinitions.Count)
                {
                    var row = grid.RowDefinitions[Grid.GetRow(value)];
                    if (row.Height.IsAbsolute)
                        row.Height = GridLength.Auto;
                }
                break;
            case Decorator decorator:
                decorator.Child = null;
                decorator.Child = wrapper;
                break;
            case ContentControl content:
                content.Content = null;
                content.Content = wrapper;
                break;
            default:
                throw new InvalidOperationException(
                    "Cannot add CSV Export below the execution summary in this XAML layout.");
        }
        wrapper.Children.Add(value);
        wrapper.Children.Add(actions);
    }

    private void InitializeTimingSections()
    {
        FrameworkElement[] keepVisible =
        [
            LatencyResultsGrid, ExecutionSummaryText,
            LatencyAverageText, LatencyMinText, LatencyMaxText,
            LatencyDatasetPathText, StartLatencyTestButton,
            LatencyExecutionModeComboBox, LatencyThresholdSlider,
            LatencyProgressText, LatencyProgressBar
        ];
        FrameworkElement[] timingControls =
        [
            FrontalLatencyGrid, FrontalSequenceCountText, FrontalMeasurementCountText,
            FrontalMinText, FrontalAverageText, FrontalMaxText,
            LateralLatencyGrid, LateralSequenceCountText, LateralMeasurementCountText,
            LateralMinText, LateralAverageText, LateralMaxText
        ];

        // Collapse the containing timing cards/columns as a whole, including headings.
        // Stop at any parent that also contains the total results or test controls.
        var sections = new HashSet<FrameworkElement>();
        foreach (var control in timingControls)
        {
            var section = control;
            while (LogicalTreeHelper.GetParent(section) is FrameworkElement parent &&
                   parent != this &&
                   !keepVisible.Any(x => IsInside(x, parent)))
                section = parent;
            sections.Add(section);
        }

        foreach (var section in sections)
        {
            if (!sections.Any(other => other != section && IsInside(section, other)))
                _timingSections.Add((section, section.Visibility));
        }
    }

    private static bool IsInside(DependencyObject child, DependencyObject ancestor)
    {
        for (DependencyObject? current = child; current is not null;
             current = LogicalTreeHelper.GetParent(current))
        {
            if (ReferenceEquals(current, ancestor))
                return true;
        }
        return false;
    }

    private void SetPerFoldTimingVisible(bool visible)
    {
        foreach (var entry in _timingColumns)
        {
            entry.Column.Width = entry.Width;
            entry.Column.MinWidth = entry.Min;
        }
        foreach (var entry in _timingRows)
        {
            entry.Row.Height = entry.Height;
            entry.Row.MinHeight = entry.Min;
        }
        _timingColumns.Clear();
        _timingRows.Clear();

        foreach (var section in _timingSections)
            section.Element.Visibility = visible ? section.Original : Visibility.Collapsed;

        if (visible)
            return;

        // If the three result sections share a Grid, release the hidden columns
        // (or rows) too, so the total results can use the available space.
        foreach (var grid in _timingSections.Select(x => LogicalTreeHelper.GetParent(x.Element))
                     .OfType<Grid>().Distinct())
        {
            var hidden = _timingSections.Where(x => LogicalTreeHelper.GetParent(x.Element) == grid)
                .Select(x => x.Element).ToArray();
            var remaining = grid.Children.OfType<UIElement>()
                .Where(x => x.Visibility != Visibility.Collapsed).ToArray();
            for (var i = 0; i < grid.ColumnDefinitions.Count; i++)
            {
                if (hidden.Any(x => Grid.GetColumn(x) <= i && i < Grid.GetColumn(x) + Grid.GetColumnSpan(x)) &&
                    !remaining.Any(x => Grid.GetColumn(x) <= i && i < Grid.GetColumn(x) + Grid.GetColumnSpan(x)))
                {
                    var column = grid.ColumnDefinitions[i];
                    _timingColumns.Add((column, column.Width, column.MinWidth));
                    column.MinWidth = 0;
                    column.Width = new GridLength(0);
                }
            }
            for (var i = 0; i < grid.RowDefinitions.Count; i++)
            {
                if (hidden.Any(x => Grid.GetRow(x) <= i && i < Grid.GetRow(x) + Grid.GetRowSpan(x)) &&
                    !remaining.Any(x => Grid.GetRow(x) <= i && i < Grid.GetRow(x) + Grid.GetRowSpan(x)))
                {
                    var row = grid.RowDefinitions[i];
                    _timingRows.Add((row, row.Height, row.MinHeight));
                    row.MinHeight = 0;
                    row.Height = new GridLength(0);
                }
            }
        }
    }
}
