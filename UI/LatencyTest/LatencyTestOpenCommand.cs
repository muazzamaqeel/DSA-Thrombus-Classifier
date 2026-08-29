using System;
using System.Windows;
using System.Windows.Input;
using UI.View;

namespace UI.LatencyTest;

public sealed class LatencyTestOpenCommand : ICommand
{
    public bool CanExecute(object? parameter) =>
        parameter is Window;

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
