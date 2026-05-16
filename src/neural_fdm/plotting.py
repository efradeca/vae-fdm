import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, window_size):
    """
    Calculate the moving average of a data series.

    Parameters
    ----------
    data: `numpy.ndarray`
        The data series to average.
    window_size: `int`
        The size of the window to average over.

    Returns
    -------
    moving_average: `numpy.ndarray`
        The moving average of the data series.
    """

    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def plot_losses(loss_history, labels):
    """
    Plot the convergence curve of a list of loss terms.

    Parameters
    ----------
    loss_history: `list` of `dict` of `float`
        The loss histories during training.
        The keys are the plot labels.
        The values are the loss values.
    labels: `list` of `str`
        The labels of the losses to plot.
    """
    # Plotting
    plt.figure(figsize=(10, 6))

    for label in labels:
        loss_values = [float(vals[label]) for vals in loss_history]
        plt.plot(loss_values, label=label)

    plt.title('Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()


def plot_smoothed_losses(loss_history, window_size, labels):
    """
    Plot the convergence curve of a list of loss terms with a moving average.

    Parameters
    ----------
    loss_history: `list` of `dict` of `float`
        The loss histories during training.
        The keys are the plot labels.
        The values are the loss values.
    window_size: `int`
        The size of the window to average over.
    labels: `list` of `str`
        The labels of the losses to plot.
    """
    # Plotting
    plt.figure(figsize=(10, 6))

    for label in labels:

        # Loss values
        loss_values = [float(vals[label]) for vals in loss_history]

        # Calculate the moving average
        smooth_loss = moving_average(loss_values, window_size)

        # Adjust the length of original loss values to match the smoothed array
        adjusted_loss_values = loss_values[:len(smooth_loss)]

        # Plot
        lines = plt.plot(adjusted_loss_values, alpha=0.5, label=label)
        color = lines[-1].get_color()
        plt.plot(smooth_loss, color=color)

    plt.title('Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()


def plot_smoothed_loss(loss_history, window_size):
    """
    Plot the convergence curve of a loss term with a moving average.

    Parameters
    ----------
    loss_history: `list` of `float`
        The loss values during training.
    window_size: `int`
        The size of the window to average over.
    """
    # Plotting
    plt.figure(figsize=(10, 6))

    # Calculate the moving average
    smooth_loss = moving_average(loss_history, window_size)

    # Adjust the length of original loss values to match the smoothed array
    adjusted_loss_values = loss_history[:len(smooth_loss)]

    # Plot
    color = "tab:blue"
    plt.plot(adjusted_loss_values, alpha=0.5, color=color)
    plt.plot(smooth_loss, color=color)

    plt.title('Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid()
    plt.show()
