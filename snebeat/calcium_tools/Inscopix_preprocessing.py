'''
Functions for loading, normalisation and splitting of calcium recording .csv-s exported from the
Inscopix Data Processing Software (IDPS).
'''

import numpy as np
import pandas as pd
from path import Path

from scipy.interpolate import interp1d
import scipy.stats as stats

import warnings


def split_recording(path):
    """
    Separates multiple recordings that have been concatenated into the same .csv file during longitudinal
    registration or time series creation by IDPS. It requires the presence of a .csv file in the same directory
    which details the names or conditions under which each recording in the concatenated file was made.
    Rows in the log file are recordings, while columns are different experimental conditions. Column titles
    can be any variable that specifies your recordings and there can be as many columns as needed, but
    there should be at least one. The number of rows should match the number of recordings in the file, otherwise
    some videos will be merged.

    Example:

    state                      test

    baseline                   resident_intruder
    baseline                   tube_test
    hungry                     resident_intruder
    hungry                     tube_test

    The name of the log file should be the same as the recording file with the addition of '_log.csv' at the end.
    """

    path = Path(path)
    directory = path.parent
    filename = path.stem

    # Load from .csv
    df = pd.read_csv(path, low_memory=False)

    # Find and load log .csv
    log_path = directory + '/' + filename + '_log.csv'
    log = pd.read_csv(log_path)

    conditions = log.columns

    # Find boundaries between concatenated videos
    timestamps = df.iloc[1:, 0].astype(float)

    frame_time_diff = timestamps.diff().mode()
    boundaries = timestamps.diff().round(decimals=3) != frame_time_diff[0].round(decimals=3)
    boundaries = np.where(boundaries)[0][1:]

    # Find duplicated log entries, merge corresponding videos and remove duplicates from log
    log_duplicates = np.where(log.duplicated(keep='first'))[0] - 1
    boundaries = np.delete(boundaries, log_duplicates)
    log.drop_duplicates(keep='first', inplace=True, ignore_index=True)

    # Merge the closest recordings if more recordings than experimental states / tests are still found, but warn user
    if len(boundaries) + 1 > log.shape[0]:
        warnings.warn(
            'There are more recordings in the file than are specified in the log file. Videos acquired closest to each other will be merged! Extend the log file to avoid this.')

        breaks = timestamps.diff()[boundaries + 1]
        merge_recordings = np.argsort(breaks)[:len(boundaries) + 1 - log.shape[0]]
        boundaries = np.delete(boundaries, merge_recordings)

    boundaries += 1

    # Separate recordings and save as .csv
    intervals = np.append(np.insert(boundaries.repeat(2), 0, [1]), df.shape[0]).reshape(-1, 2)

    for n, interval in enumerate(intervals):

        save_path = directory + '/' + filename

        for condition in conditions:
            save_path += '_' + log.loc[n, condition]

        recording = df.iloc[interval[0]:interval[1], :]
        recording.loc[-1] = df.loc[0]    # This line is throwing SettingWithCopy warning from pandas
        recording.index = recording.index + 1
        recording.sort_index(inplace=True)
        recording.to_csv(save_path + '.csv', index=False)

    print('Split complete')


def load_inscopix(path, normalisation='z-score'):
    """
    Loads neuron activity traces from an Inscopix .csv result file, selects only accepted cells, and optionally normalises
    each cell to itself to interval [0,1] (fixed interval normalisation) or by z-scoring the entire recording.
    This function is not suitable for loading files containing spike data.

    Args:
        path: str; path to Inscopix result .csv file
        normalisation: str; type of normalisation to run on the data. ['z-score', 'fixed interval']. If None no normalisation
                            will be performed.

    Returns:
        df: np.array; an NxM matrix, where N are cells and M are timepoints in the recording
    """

    df = pd.read_csv(path, low_memory=False)

    df = df.iloc[:, 1:]
    df = df.loc[1:, df.iloc[0] == ' accepted']
    df = np.array(df).astype(float)

    if normalisation is not None:

        if normalisation == 'z-score':
            df = stats.zscore(df, axis=0)

        else:
            for n, cell in enumerate(df.T):
                f = interp1d([cell.min(), cell.max()], [0, 1])
                cell = f(cell)

                df[:, n] = cell

    return df.T
