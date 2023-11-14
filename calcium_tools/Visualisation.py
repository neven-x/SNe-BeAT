import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def tuning_barplot(tuning_scores, color='C0', ax=None):
    if ax is None:
        ax = plt.gca()

    ## Set axis properties
    ax.yaxis.set_visible(False)
    ax.set_ylim(0, len(tuning_scores))
    plt.gca().invert_yaxis()
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.barh(np.arange(0.5, len(tuning_scores) + 0.5, 1), tuning_scores, 0, color=color)
    ax.axvline(0, c='black', linestyle='--')

    ax.set_xticks(np.arange(-1, 1, 0.5))
    ax.set_xlabel('Tuning index: ' + tuning_scores.name)

    return ax


def plot_data(df, behavs, events, title=None, fr=10):
    """

    Plots analysed Inscopix data with behaviour.

    Args
        df: A 2D array containing the calcium data of shape NxM, where N are cells and M are samples.

        behavs: A 1D array of the names of all behaviours in the recording

        events: A list of event arrays for each behaviour (used for the ethogram)

        tuning_scores: a Pandas Series (with a name attribute) of behavioural tuning scores for each cell
                      (must be the same length as number of rows in df)

        title: str; The title of the figure

        fr: int; fps of the calcium recording
    """

    plt.rcParams.update({'font.size': 18,
                         'lines.linewidth': 2})

    # Setting up figure
    fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [1.5, 5]}, figsize=[20, 7])
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.01)

    # Plotting the behaviour raster plot
    colors = ['C{}'.format(i) for i in range(len(events))]
    axs[0].eventplot(events, colors=colors, linelengths=0.8)

    if title is not None:
        axs[0].set_title(title)

    axs[0].set_xlim(0, df.shape[1])
    axs[0].set_frame_on(False)
    axs[0].xaxis.set_visible(False)

    axs[0].set_yticks(range(len(behavs)))
    axs[0].yaxis.set_tick_params(length=0)
    axs[0].set_yticklabels(behavs)
    for ytick, color in zip(axs[0].get_yticklabels(), colors):
        ytick.set_color(color)

    # Plotting the calcium data
    norm = TwoSlopeNorm(vcenter=0)
    img = axs[1].imshow(df, aspect='auto', cmap='bwr', norm=norm)
    axs[1].set_frame_on(False)
    axs[1].set_xticks(np.arange(0, df.shape[1], 60 * fr))
    axs[1].set_xticklabels(np.arange(0, df.shape[1], 60 * fr) // (60 * fr))
    axs[1].set_xlabel('Time (min)')
    axs[1].set_ylabel('Cell #')

    return fig


def paired_data_lineplot(data, conditions, marker_colors, line_color, label=None, xlim=None, ylim=None, xlabel=None,
                         ylabel=None, ax=None):
    """
    Args:
        data: N pairs by M groups
        conditions: list; List of experimental condition names for the x-axis (strings)
        marker_colors: dict: keys are groups (as integer numbers), values are colours
        line_color: line color

    Returns:
        ax: plt.axes object
    """

    if ax is None:
        ax = plt.gca()

    for n, group in enumerate(data.T):
        plt.plot(data.T, color=line_color, alpha=1, label=label, zorder=1)
        ax.scatter(x=[n] * len(group), y=group, color=marker_colors[n], zorder=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(conditions)
    ax.xaxis.set_tick_params(length=0)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return ax
