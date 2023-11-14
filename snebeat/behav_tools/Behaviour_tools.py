import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.interpolate import interp1d


def preprocess_behaviour(path, inscopix_len=None):
    """
    Imports behaviour data from BORIS .csv file and constructs a binary ON/OFF array for each behaviour.

    Args
        path: string; path to the behaviour data .csv file.
        inscopix_len: None, int; the number of frames in the corresponding Inscopix recording (processed not original).
                    If None the number of frames in the behaviour recording will be used.

    Returns
        num_episodes: Number of behaviour episodes detected
        bdf_p: A pd.DataFrame with binary behaviour arrays where each column is a different behaviour
        events: A list of behaviour event arrays for creating ethograms
        fps: frame rate of the behaviour video
    """

    # Import behaviour data
    bdf = pd.read_csv(path)

    header = np.where(bdf.iloc[:, 0] == 'Time')[0][0]
    bdf.columns = bdf.iloc[header]
    bdf = bdf.drop(range(header + 1))
    bdf = bdf.reset_index()

    fps = int(float(bdf['FPS'][0]))

    # Check if argument type is correct
    if inscopix_len is None:
        duration = int(float(bdf['Total length'][0]))
        inscopix_len = duration * fps

    elif type(inscopix_len) is not int:
        raise TypeError("inscopix_len has to be an integer")

    # Interpolate from behaviour times to Inscopix frames
    f = interp1d([0, float(bdf['Total length'][0])], [0, inscopix_len - 1])
    bdf['Time'] = f(bdf['Time'].astype(float))

    # Extract behaviour names
    behavs = bdf['Behavior'].unique()

    # Initialise empty results variable
    num_episodes = pd.DataFrame()
    bdf_p = pd.DataFrame()
    events = []

    # Construct a binary ON/OFF array for each behaviour
    for n, behav in enumerate(behavs):
        behav = bdf.loc[bdf['Behavior'] == behav, :]

        # Extract behaviour start and stop indices expressed in inscopix frames
        starts = behav['Time'][behav['Status'] == 'START']
        starts = starts.astype(int)

        stops = behav['Time'][behav['Status'] == 'STOP']
        stops = stops.astype(int)

        binary = np.zeros(inscopix_len)

        # Make binary array and store it in the result variable
        if starts.empty:
            point = behav['Time'][behav['Status'] == 'POINT']
            binary[point.astype(int)] = 1

        else:
            for episode in zip(starts, stops):
                start = episode[0]
                stop = episode[1]

                binary[start:stop] = 1

        num_episodes.loc[0, behavs[n]] = len(starts)
        bdf_p[behavs[n]] = binary
        bdf_p = bdf_p.astype(int)

        events.append(binary.nonzero()[0])

    return num_episodes, bdf_p, events, fps


def behaviour_metrics(bdf, reference=None, window=60, fr=10):
    """
    bdf: pd.df
    reference: string; (optional) a reference behaviour from whose first episode the quantification window will start.
                        If None, the start of the dataset will be used.
    window: int; window (in seconds) in which to quantify behavioural metrics. If None, the entire dataset is quantified.
    fr: int; framerate in fps

    Returns
    normalised_duration: dict; percentage of time engaged in each behaviour
    episode_frequency: dict; episode frequency per minute for each behaviour
    episode_length: dict; the median episode duration in seconds for each behaviour
    latency: dict; latency to first instance of each behaviour in seconds
    """

    if reference is None:
        start = 0
    else:
        start = np.where(bdf[reference] == 1)[0][0]

    if window is None:
        end = bdf.shape[0]
    else:
        end = window * fr + start

    normalised_duration = {}
    episode_frequency = {}
    episode_length = {}
    latency = {}
    for behav in bdf.columns:

        data = bdf[behav].loc[start:end]

        normalised_duration[behav] = data.sum() / len(data) * 100

        starts = np.where(np.array(data)[1:] > np.array(data)[:-1])[0]
        ends = np.where(np.array(data)[1:] < np.array(data)[:-1])[0]

        if len(ends) > len(starts):
            ends = ends[1:]
        elif len(starts) > len(ends):
            starts = starts[:-1]
        elif all((ends - starts) < 0):
            starts = starts[:-1]
            ends = ends[1:]

        try:
            latency[behav] = starts[0] / fr
        except IndexError:
            continue

        episode_frequency[behav] = len(starts) / ((end - start) / fr / 60)

        episode_length[behav] = np.median((ends - starts) / fr)

    return normalised_duration, episode_frequency, episode_length, latency


def combine_behavs(bdf, behavs, name):
    ''' Takes the binary arrays for 2 or more existing behaviours in bdf and merges them to create a new
    combined behaviour.'''

    """
    bdf: pd.DataFrame
    behavs: list; list of behaviour names from bdf to combine (strings)
    name: string; name of the new combined behaviour

    Returns
    new_bdf: a copy of bdf with new behaviour addes as a new column
    """

    new_behav = np.zeros(bdf.shape[0])

    new_behav[bdf[behavs].sum(axis=1) > 0] = 1

    new_bdf = bdf.copy()
    new_bdf[name] = new_behav.astype(int)

    return new_bdf


def segment(bdf, start_ref = None, stop_ref = None, state_ref = None, pad = None, reindex=True):
    '''
    Splits behavioural recording into multiple segments / trials, based on a landmark behavioural annotation e.g. introduction of an intruder, presentation of a stimulus etc.

    :param bdf:
    :param start_ref: a point event that defines the start of each segment
    :param stop_ref: (optional) a stop event that defines the end of each segment. If None, each segment runs until the
                        start of the next segment or the end of the recording in case of the last segment.
    :param state_ref: a state event that defines each segment; if both state and point references are passed, the state
                        reference will be used
    :pad: optionally specifies the number of padding frames added before and after each extracted segment
    :param reindex: bool; specifies whether the new behaviour dataframes are reindexed to start from 0 instead of their
                        old index.

    :return:
    segments: list; a list of behaviour recording segments / trials
    boundaries: list; list of start and end index pairs for each segment
    '''

    if state_ref is not None:
        starts = np.where(bdf[state_ref].diff() > 0)[0]
        stops = np.where(bdf[state_ref].diff() < 0)[0] - 1

    else:
        starts = np.where(bdf[start_ref])[0]

        if stop_ref is None:
            stops = np.append(starts[1:], bdf.shape[0])
        elif start_ref == stop_ref:
            starts = starts[0::2]
            stops = starts[1::2]
        else:
            stops = np.where(bdf[stop_ref])[0]

    if pad is not None:
        starts -= pad
        stops += pad

    segments = []
    boundaries = list(zip(starts, stops))
    for bounds in boundaries:
        start, stop = bounds

        segment = bdf.loc[start:stop]
        if reindex:
            segment.reset_index(inplace=True, drop=True)

        segments.append(segment)

    return segments, boundaries

def behaviour_subset(binary, n):
    """Randomly selects a subset of n behavioural episodes from binary behaviour array

    Args:
        binary: np.array; a binary behaviour array, where presence of behaviour is 1 and absence is 0
        n: int, number of behavioural episodes to select from binary

    Returns:
        subset: np.array; an array of length == len(binary) but containing only the selected subset of behavioural episodes

    """

    all_starts = np.where(binary.diff() > 0)[0]
    all_stops = np.where(binary.diff() < 0)[0]

    if len(all_starts) < n:
        raise ValueError('The behaviour subset must be smaller than the total number of behavioural episodes.')

    selected = np.random.choice(range(len(all_starts)), n, replace=False)

    starts = all_starts[selected]
    stops = all_stops[selected]

    subset = np.zeros(len(binary))
    for episode in zip(starts, stops):
        start = episode[0]
        stop = episode[1]

        subset[start:stop] = 1

    return subset
