import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_behav_episodes(df, bdf, behav, window, separation=None, fr=10, subset=None, standardised=False):
    """
    Extracts segments of the calcium recording corresponding to a user specified window around the start or end of a behavioural episode.

    Args:
        df: np.array; The calcium recording - an MxP matrix, where M are cells and P are timepoints in the recording
        bdf: pd.DataFrame; a pandas dataframe containing binary behaviour arrays in columns. Column names must be behaviour names.
        behav: str; specifies which behaviour to extract
        window: float; the recording duration before and after behavioural events in seconds that you would like to extract. The final extracted episode will be twice the window duration.
        separation: float, the minimum temporal separation in seconds between an episode and the end of the previous event
        fr: int; framerate of the recording
        subset: tuple or 2-element list/array; Specify the start and end index (in frames) if you wish to extract episodes from
                                                a smaller temporal window of the recording
        standardised: bool; Specifies whether each episode is standardised

    Returns:
        episodes: np.array; an NxMxP matrix, where N is the number of instances of the specified behaviour, M is the number of neurons in df (the recording) and P is the window in frames
    """

    # Convert window from seconds to frames
    window *= fr
    window = int(window)

    # Identify behavioural events
    starts = np.where(bdf[behav].diff() > 0)[0]
    stops = np.where(bdf[behav].diff() < 0)[0]

    # Exclude events that are not sufficiently separated in time from neighbouring events
    if separation is not None:
        separation *= fr
        separation = int(separation)

        excluded = np.where(np.abs(starts[1:] - stops[:-1]) < separation)[0]
        starts = np.delete(starts, excluded)

    # Exclude events which are outside the specified subset window of the recording
    if subset is not None:
        excluded = np.array([], dtype=int)
        early_limit, late_limit = subset

        excluded = np.append(excluded, np.where(starts < early_limit)[0])
        excluded = np.append(excluded, np.where(starts > late_limit)[0])

        starts = np.delete(starts, excluded)

    # Exclude behavioural events whose window will reach over the limits of the calcium dataframe
    starts = np.delete(starts, (starts + window) >= df.shape[1])
    starts = np.delete(starts, (starts - window) < 0)

    # Prepare result variable
    episodes = np.zeros((len(starts), int(df.shape[0]), window * 2))

    # For each behavioural episode extract corresponding calcium activity
    for n, event in enumerate(starts):
        episode = np.arange(event - window, event + window)

        if episode[-1] > df.shape[1]:
            continue
        else:
            episode = df[:, episode]

            if standardised:
                scaler = StandardScaler()
                scaler.fit(episode)
                episode = scaler.transform(episode)

            episodes[n] = episode

    return episodes


def behaviour_episode_zscore(data, baseline_interval):
    """
    Z scores an array of behaviour episodes using a user-defined baseline period for mean and standard deviation calculation

    Args
        data: np.array; PxNxM matrix, where P are episodes, N are cells and M are timepoints in the behavioural episode.
        baseline_interval: list; [start, end] specifies the start and end indices (in frames) of the baseline interval

    Returns
        zscore: np.array, zscored version of data with the same shape
    """

    bstart, bend = baseline_interval

    if type(bstart) != int or type(bend) != int:
        raise TypeError('baseline_interval can only contain integers.')

    baseline = data[:, :, bstart:bend]
    baseline_mean = baseline.mean(axis=2)
    baseline_std = np.std(baseline, axis=2)

    zscores = np.moveaxis(data, [2], [0]) - baseline_mean
    zscores = zscores / baseline_std
    zscores = np.moveaxis(zscores, [0], [2])

    return zscores


def sort_episodes_by_magnitude(episodes, interval=None, absolute=True):
    '''
    Take a list of PSTH episodes, quantifies the neural response magnitude for each episode and sorts the episodes
    in descending order of response magnitude.

    Args
        episodes: numpy matrix with shape NxMxP, where N is number of episodes, M is number of neurons and P is number of timepoints in each PTSH.
        interval: 2-elements tuple containing the start and end indices of the time interval within the episode for which response magnitude is calculated.
                    If None, the response part of the PTSH (the second half) is used by default.
        absolute: bool; specifices whether absolute responses are considered. Setting to false will find episodes with the most excitatory
                        response across the population.

    Returns
        sorted_episodes: Same format as input episodes with first dimension sorted in descending order of response magnitude across
        all neurons
    '''

    if interval is None:
        summed_episode_activity = episodes[:, :, episodes.shape[2]//2:]
    else:
        summed_episode_activity = episodes[:, :, interval[0]:interval[1]]

    if absolute:
        summed_episode_activity = np.abs(summed_episode_activity)

    summed_episode_activity = summed_episode_activity.sum(axis=(1, 2))
    new_order = np.argsort(-summed_episode_activity)
    sorted_episodes = episodes[new_order]

    return sorted_episodes


def segment_calcium_recording(df, boundaries):
    '''
    df: A 2D matrix of cells x timepoint
    boundaries: list of pairs that specify the start and stop indices of each segment

    Returns
    new_df: A processed version of the input df, where only the specificed segments have been kept and concatenated
            into a new recording.s
    '''

    df_segments = []
    for segment in boundaries:
        start, end = segment

        df_segment = df[:, start:end+1]
        df_segments.append(df_segment)

    new_df = np.hstack(df_segments)

    return new_df
