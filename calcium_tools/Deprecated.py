import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import scipy.stats as stats

from warnings import warn
import warnings


def estimate_tuning_global(df, binary, tuned_to, history=1, cv=8, scale_auc=True, remove_nontarget=True,
                           subtract_shuffled_tuning=True):
    """
    Estimates the tuning of each cell in dataframe df to specified binary variable by fitting a logistic GLM. Time lags of each
    data point can optionally be used as additional regressors. Tuning is expressed as a 6-fold cross-validated ROC AUC.

    Args
        df: A 2D array of shape NxM, where N are cells and M are samples
        binary: pd.DataFrame; columns are binary ethogram arrays and have string names
        tuned_to: str; name of the behaviour contained in binary for which tuning will be estimated
        history: int, the number of time lags to use as additional regressors. history = 1 does not use any history data.
        cv: int; specifies number of data splits for cross-validation
        scale_auc: bool; Sets ROC AUC values below 0.5 to 0 and scales the remaining values to an index between 0 and 1.
        remove_nontarget: bool; specifies whether data where non-target behaviours (other than tuned_to) are ocurring is removed
                                from the baseline fluorescence dataset, since keeping it can cause tuning to the target to be underestimated
        subtract_shuffled_tuning: bool; Specifies if the tuning index obtained from shuffled calcium data should be subtracted
                                        from the tuning index estimated from true data.

    Returns:
        aucs: an array of ROC AUC values for each cell
    """

    # Find timepoints when non-target behaviours are ocurring so they can be excluded from the baseline
    if remove_nontarget:

        exceptions = []

        if tuned_to in ('push', 'retreat', 'resist'):
            exceptions = ['tube test']

        if tuned_to in ('sniff', 'sniff AG'):
            exceptions = ['chemoinvestigation']

        if tuned_to == 'chemoinvestigation':
            exceptions = ['sniff', 'sniff AG']

        for exception in exceptions:
            if exception not in binary.columns:
                exceptions.remove(exception)

        exceptions.append(tuned_to)

        included = ~binary[binary.columns.drop(exceptions)].sum(axis=1).astype(bool)
    else:
        included = [True] * df.shape[1]

    # Find the binary array for the desired behaviour and remove indices where non-target behaviours are ocurring
    tuned_to = binary[tuned_to][included]

    # Define model
    model = LogisticRegression(solver='saga', penalty='l2', max_iter=2500)

    # Prepare empty result list
    aucs = []

    # Loop over cells in the recording
    for cell in df:

        # Remove cell activity data where non-target behaviours are ocurring
        cell = cell[included]

        # Make design matrix
        X = make_design_matrix(cell, history)

        # 8-fold cross-validate the model on data
        model_score = cross_val_score(model, X, tuned_to, scoring='roc_auc', cv=cv).mean()

        # Scale AUC scores to interval from 0 to 1
        if scale_auc:

            model_score = (model_score - 0.5) * 2
            if model_score < 0: model_score = 0

        # Estimate tuning to a shuffled calcium dataset
        if subtract_shuffled_tuning:

            shuffled_cell = cell.copy()
            np.random.shuffle(shuffled_cell)

            shuffled_X = make_design_matrix(shuffled_cell, history)

            shuffled_score = cross_val_score(model, shuffled_X, tuned_to, scoring='roc_auc', cv=cv).mean()

            # Scale AUC scores to interval from 0 to 1
            if scale_auc:

                shuffled_score = (shuffled_score - 0.5) * 2
                if shuffled_score < 0: shuffled_score = 0

            model_score -= shuffled_score

            if model_score < 0: model_score = 0

        # Determine if tuning is positive or negative and represent that in sign of the AUC score
        if cell[tuned_to != 0].mean() < cell[tuned_to == 0].mean():
            model_score *= -1

        # Append result to return variable
        aucs.append(model_score)

    aucs = np.array(aucs)

    return aucs


def estimate_tuning_episodewise(episodes, event_index=None, history=1, cv=4):
    """
    Calculates cell behavioural tuning based on behavioural episode data - temporal windows around a behavioural event.
    Uses a logistic GLM to model the change in activity from baseline state to behavioural state.

    Args:
        episodes: An IxJxK matrix, where I are episodes, J are cells and K are timepoints in the behavioural episode
        event_index: int; Index in a behavioural episode window which separates
        history: int, the number of time lags to use as additional regressors. history = 1 does not use any history data.
        cv: int, specifies fold crossvalidation.

    Returns:
        aucs: an array of ROC AUC values for each cell
    """

    # Define model
    model = LogisticRegression(solver='saga', penalty='l2', max_iter=2500)

    if event_index is None:
        event_index = episodes.shape[2] // 2

    # Pool baseline and signal data across different episodes
    baseline = episodes[:, :, :event_index]
    baseline = np.concatenate(baseline, axis=1)
    signal = episodes[:, :, event_index:]
    signal = np.concatenate(signal, axis=1)

    # Define the encoding variable which splits each episode into the baseline and signal (behaviour) dataset.
    index = np.concatenate((np.zeros(baseline.shape[1]), np.ones(signal.shape[1])))
    index = index.astype(bool)

    def normalise_auc(auc):
        auc = (auc - 0.5) * 2
        if auc < 0:
            auc = 0

        return auc

    # Prepare empty output variable
    aucs = np.zeros(episodes.shape[1]) * np.nan

    for j, cell in enumerate(zip(baseline, signal)):

        cell = np.concatenate((cell[0], cell[1]))

        # Reformat data into design matrix and create a shuffled calcium dataset
        X = make_design_matrix(cell, history=history)

        shuffled_cell = cell.copy()
        np.random.shuffle(shuffled_cell)
        X_shuffled = make_design_matrix(shuffled_cell, history=history)

        # 8-fold cross-validate model on real and shuffled data
        model_score = cross_val_score(model, X, index, scoring='roc_auc', cv=cv).mean()
        shuffled_score = cross_val_score(model, X_shuffled, index, scoring='roc_auc', cv=cv).mean()

        # Scales aucs to interval between 0 and 1. Scores below chance rate are set to 0.
        model_score = normalise_auc(model_score)
        shuffled_score = normalise_auc(shuffled_score)

        # Subtract shuffled score from actual score
        model_score -= shuffled_score
        if model_score < 0:
            model_score = 0

        # Determine if tuning is positive or negative and represent that in sign of the AUC score
        if cell[index].mean() < cell[~index].mean():
            model_score *= -1

        aucs[j] = model_score

    return aucs


def test_tuning_distribution_differences(tuning_mouse1, tuning_mouse2, alternative='two-sided'):
    """

    Tests for differences in the distribution of tuning scores for all (common) behaviours between two mice using the
    Kolmogorov-Smirnov test.

    Args
        tuning_mouse1 & tuning_mouse2: pd.DataFrames containing behaviour labelled tuning scores for each cell
        alternative: str; specifies the alternative hypothesis of the KS test, consult scipy.stats.kstest docs for details

    Returns
        difference: dict; a dictionary of KS p-values for each behaviour type
    """

    differences = {}
    for behav in tuning_mouse1.columns:
        try:
            _, p_value = stats.kstest(tuning_mouse1[behav], tuning_mouse2[behav], alternative=alternative)
            differences[behav] = p_value
        except:
            continue

    return differences


def tuning_proportions(aucs, untuned_boundary=0):
    """
    Args:
        aucs: dict / pd.DataFrame: binary behaviour arrays
        untuned_boundary: float: tuning value between which a neuron is considered untuned

    Returns:
        results: dict; a dictionary of tuples containing tuning proportions for each behaviour.
                        The first, second and third array element contain the proportion of positively tuned,
                        untuned and negatively tuned cells respectively.
    """

    results = {}
    for behav in aucs.columns:
        # Proportion of positively tuned cells
        positive = len(np.where(aucs[behav] > untuned_boundary)[0]) / aucs.shape[0]

        # Proportion of negatively tuned cells
        negative = len(np.where(aucs[behav] < -untuned_boundary)[0]) / aucs.shape[0]

        # Proportion of untuned cells
        untuned = 1 - positive - negative

        results[behav] = np.array([positive, untuned, negative])

    return results


def analyse_recording(path, cage, rank, state, test, plotting=True, custom_behaviour=None, average_estimates=False):
    """
    Loads calcium and behavioural data, estimates behavioural tuning for each cell in the recording and for each
    behaviour. Optionally plots the calcium, behavioural and tuning data.

    Args
        path: str; path to directory with recordings
        cage: str; cage number
        rank: str; 'dom', 'sub'
        state: str; 'baseline', 'rising', 'defeated'
        test: str; 'TT', 'RI_male', 'RI_female'
        plotting: bool; specifies if data should be plotted
        custom_behaviour: str; specifies a user defined behaviour for which tuning will be visualised in the plots
        average_estimates: bool, specifies whether tuning estimates are repeated and averaged over several random
                                 subsets of behavioural data. Recommended where the number of behavioural episodes
                                 is low (i.e. less than 10). Can significantly slow down performance.

    Returns
        tuning: pd.DataFrame containing behaviour labelled tuning scores for each cell
    """

    # Adds a slash to end of the path if missing
    if path[-1] != '/':
        path += '/'

    filename = f'{cage}_{rank}_{state}_{test}.csv'

    # Load inscopix data
    df = load_inscopix(f'{path}Inscopix/{filename}', normalisation='fixed interval')

    # Load and process behaviour data
    num_episodes, bdf, events = preprocess_behaviour(f'{path}Behaviour/Inscopix/{filename}', df.shape[1])

    # If multiple types of sniffing annotated, create a merged chemoinvestigation behaviour
    try:
        bdf['chemoinvestigation'] = bdf['sniff'] + bdf['sniff AG']
        bdf['chemoinvestigation'] = bdf['chemoinvestigation'].where(~(bdf['chemoinvestigation'] > 1), other=1)
    except KeyError:
        pass

    # Estimate tuning to all behaviours
    tuning = {}
    for behav in bdf.columns:

        # Skip behaviour if behaviour is invalid or not enough data for 8-fold cross-validation
        if behav == 'invalid':
            continue

        elif bdf[behav].sum() < 8:
            continue

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # If averaging, estimate tuning on 10 subsets of the behavioural data
                if average_estimates:
                    try:
                        samples = np.arange(num_episodes[behav][0], dtype=int)[-10:] + 1
                    except:
                        samples = np.arange(num_episodes[behav][0], dtype=int) + 1

                    aucs = np.zeros((df.shape[0], len(samples)))
                    for n, subset in enumerate(samples):
                        subset = behaviour_subset(bdf[behav], subset)
                        subset = pd.DataFrame(subset, columns=[behav])
                        subset = bdf.copy()[behav] = subset

                        aucs[:, n] = estimate_tuning_global(df, subset, behav, history=1, remove_nontarget=True)

                    aucs = aucs.mean(axis=1)

                else:
                    aucs = estimate_tuning_global(df, bdf, behav, history=1, remove_nontarget=True)

                # Check if valid score was produced
                if np.isnan(aucs).sum() > 0:
                    continue
                else:
                    tuning[behav] = aucs

    tuning = pd.DataFrame(tuning)

    # Plotting
    if plotting:
        # If no custom behaviour, find the most strongly tuned behaviour
        if custom_behaviour is None:
            strongest = tuning.columns[tuning.abs().mean().argmax()]

        else:
            strongest = custom_behaviour

        # Sort cells based on tuning similarity to the most strongly tuned behaviour
        sort_index = np.flip(np.argsort(tuning[strongest]))
        tuning.sort_values(by=strongest, ascending=False, inplace=True)
        df = df[sort_index]

        # Convert recording identifiers to presentable text
        presentable_ranks = {'dom': 'Dominant',
                             'sub': 'Subordinate'}

        presentable_tests = {'TT': 'Tube test',
                             'RI_male': 'Male intruder',
                             'RI_female': 'Female intruder'}

        title = presentable_ranks[rank] + ' - ' + presentable_tests[test]

        fig = plot_data(df, bdf.columns, events, tuning[strongest], title)

    return tuning


def analyse_recording_episodewise(path, cage, rank, state, test, window, fr=10, num_episodes=None, separation=None,
                                  custom_behaviour=None, plotting=True):
    """
    Loads calcium and behavioural data, estimates behavioural tuning episodewise for each cell in the recording and for each
    behaviour. Optionally plots the calcium, behavioural and tuning data.

    Args:
        path: str; path to directory with recordings. Required directory structure below:
                    path ---> Behaviour/Inscopix/Behaviour.csv
                      |
                      |-----> Inscopix/Calcium.csv
        cage: str; cage number
        rank: str; 'dom', 'sub'
        state: str; 'baseline', 'rising', 'defeated'
        test: str; 'TT', 'RI_male', 'RI_female'
        window: float; window around each behavioural event in seconds that is used for tuning analysis
                        e.g. window = 3 would define a 3-second baseline period before the event and a 3-second
                        signal period after the behavioural event
        fr: int; calcium recording frame rate
        num_episodes: dict; Specifies the number of behavioural episodes used to estimate tuning for particular behaviour
                            dict keys are behaviour labels and values are integers specifying the number of episodes used.
                            By default, all episodes in the recording are used. This is not recommended if you wish to
                            compare recordings with different numbers of behavioural episodes.
        separation: float; minimum temporal distance (in seconds) of the start of an episodes from the end of the preceding episode
                    for it to be considered tuning calculation.
        custom_behaviour: str; specifies a user defined behaviour for which tuning will be visualised in the plots
        plotting: bool; specifies if data should be plotted

    Returns:
        tuning: pd.DataFrame of tuning values where rows are cells and columns are behaviours.
    """

    # Load calcium and behavioural data
    df = load_inscopix(path + f'Inscopix/{cage}_{rank}_{state}_{test}.csv', normalisation='fixed interval')
    _, bdf, events = preprocess_behaviour(path + f'Behaviour/Inscopix/{cage}_{rank}_{state}_{test}.csv', df.shape[1])

    # Keep only valid behaviours
    if 'invalid' in bdf.columns:
        index = np.where(bdf.columns == 'invalid')[0][0]
        bdf.drop(columns=bdf.columns[index], inplace=True)
        del events[index]

    # Merges different types of sniffing into a 'chemoinvestigation' behaviour
    try:
        bdf['chemoinvestigation'] = bdf['sniff'] + bdf['sniff AG']
        bdf['chemoinvestigation'] = bdf['chemoinvestigation'].where(~(bdf['chemoinvestigation'] > 1), other=1)
    except:
        pass

    # Estimate tuning for different behaviours
    tuning = {}
    for behav in bdf.columns:

        # Extract calcium data surrounding behavioural events
        episodes = extract_behav_episodes(df, bdf, behav, window=window, separation=separation, fr=fr)

        # Restrict the number of episodes to a subset specified in num_episodes
        if num_episodes is not None:
            if behav in num_episodes.keys():
                sample = num_episodes[behav]
                if episodes.shape[0] < sample:
                    warn(f'Not enough {behav} episodes in cage {cage}_{rank}_{state}_{test}')
                    continue
                episodes = episodes[:sample]
            else:
                warn(f'Number of episodes used for tuning estimation is not specified for behaviour "{behav}".')
                continue

        episodes = behaviour_episode_zscore(episodes, [0, int(window * fr)])
        aucs = estimate_tuning_episodewise(episodes)

        # Check if valid score was produced
        if np.isnan(aucs).sum() > 0:
            continue
        else:
            tuning[behav] = aucs

        tuning[behav] = aucs

    tuning = pd.DataFrame(tuning)

    # Plotting
    if plotting:
        # If no custom behaviour, find the most strongly tuned behaviour
        if custom_behaviour is None:
            strongest = tuning.columns[tuning.abs().mean().argmax()]

        else:
            strongest = custom_behaviour

        # Sort cells based on tuning similarity to the most strongly tuned behaviour
        sort_index = np.flip(np.argsort(tuning[strongest]))
        tuning.sort_values(by=strongest, ascending=False, inplace=True)
        df = df[sort_index]

        # Convert recording identifiers to presentable text
        presentable_ranks = {'dom': 'Dominant',
                             'sub': 'Subordinate'}

        presentable_tests = {'TT': 'Tube test',
                             'RI_male': 'Male intruder',
                             'RI_female': 'Female intruder',
                             'RI_marathon': 'Resident-intruder marathon',
                             'TT_marathon': 'Tube-test marathon'}

        title = presentable_ranks[rank] + ' - ' + presentable_tests[test]

        fig = plot_data(df, bdf.columns, events, tuning[strongest], title)

    return tuning, fig
