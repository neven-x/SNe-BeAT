''' A series of functions for calculating Elo scores and ranking agents competing in a games with binary win/lose outcomes
and tools for evaluating the stability of the ranking.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from path import Path
from scipy.stats import rankdata, chi2_contingency, chisquare
from scipy.interpolate import interp1d
from pprint import pprint
import warnings
from Utils import concat_dicts

def predicted_prob(Ra, Rb):

    rank_diff = Rb - Ra
    probability = 1/(1 + 10**(rank_diff/400))

    return probability


def update_elo(Ra_old, Rb_old, result, k=100):
    # Ra_old, Rb_old: existing rating of players A and B to be updated
    #
    # Result: integer or float indicating game outcome for player A
    # 1 for win
    # 0.5 for draw
    # 0 for loss
    #
    # k = integer constant indicating the maximum number of points transferred between players in one game

    if result not in [0, 0.5, 1]:
        raise ValueError('Result must be either 1 (victory), 0.5 (draw) or 0 (loss).')

    Pa = predicted_prob(Ra_old, Rb_old)

    transfer = k*(result - Pa)

    Ra_new = Ra_old + transfer
    Rb_new = Rb_old - transfer

    return Ra_new, Rb_new

def stability_index(ranks, exclude_first=True):
    """
    Calculates the hierarchy stability index modified from Neuman et al. 2011 over a 2 day sliding window
    S ranges between 0 (completely stable hierarchy) and 1 - large rank reversals every day - complete instability

    Does not work with unequal number of players on different days

    ranks = a dictionary of ranks or other scores (e.g. Elo ratings) with animals as keys
    and their ratings dictionary values formatted as lists/arrays where every element represents the rating on a
    particular date
    """

    ranks = pd.DataFrame(ranks)
    S = []

    for day in range(len(ranks) - 1):
        # Calculates standardised Elo ratings
        rank_window = ranks.iloc[day:day+2]
        best = rank_window.iloc[0].max()
        worst = rank_window.iloc[0].min()
        middle = np.median(rank_window.iloc[0])
        deviation = abs(rank_window.iloc[0] - middle)
        f = interp1d([deviation.min(), deviation.max()], [0,1])

        deviation = abs(rank_window.iloc[0] - middle)
        standard_elo = f(deviation)

        # Calculates rank changes over a 2 day sliding window and scales by standardised Elo rating
        d1,d2 = rank_window.values
        rank_change = abs(d2 - d1)*standard_elo
        stability_index = sum(rank_change)
        stability_index /= (len(rank_window.columns)*2)

        S.append(stability_index)

    if not exclude_first:
        return S
    else:
        return 1 - np.array(S[1:])


def consistency(df):
    """
    Compares game outcomes between the same 2 animals over 2 consecutive days as a rolling window and returns a
    consistency score for each day (except the first day where consistency comparison is not possible)

    Consistency score ranges between 0 (all game outcomes between all animal pairs were inconsistent across 2 days)
    and 1 (all game outcomes were consistent across 2 days). The score is an average of the consistency score for each
    animal pair

    df = pandas dataframe containing a record of tests run on each date and the winner + loser of each test

    Returns:

    consistency = A list of consistency scores for each day (except the first)
    incosistency_log = a dictionary of the dates and animal pairs where outcomes were inconsistent
    """

    consistency = []
    inconsistency_log = {}

    dates = np.unique(df['Date'])

    for day in range(1, len(dates)):

        df_window = df[np.logical_or(df['Date'] == dates[day], df['Date'] == dates[day - 1])]
        num_games = df_window.shape[0]

        names = np.unique(df_window[['Winner', 'Loser']])

        combinations = list(itertools.combinations(names, 2))
        combinations = [combination[0] + combination[1] for combination in combinations]

        # Assigns each game to an animal pair
        pairs = []

        for n in range(num_games):
            winner, loser = df_window.iloc[n][['Winner', 'Loser']]

            for combination in combinations:
                match = winner in combination and loser in combination

                if match:
                    pairs.append(combination)

        if len(pairs) != df_window.shape[0]:
            raise Warning('Error in test records for date %s' % dates[day])

        # Loops over animal pairs, finds game outcomes between these animals and compares if the outcomes are the same

        consistency_index = np.zeros(len(combinations))
        found = False

        for n, combination in enumerate(combinations):

            replicates = df_window[np.array(pairs) == combination]
            nans = 0

            if replicates.shape[0] == 2:
                if replicates.iloc[0]['Winner'] != replicates.iloc[1]['Winner']:

                    consistency_index[n] = 1
                    found = True

            else:
                Warning('Test between animals %s and %s missing for date %s or %s' % (combination[0], combination[1], dates[day - 1], dates[day]))
                nans += 1

        # If inconsistent outcome is found, add the date and animal pair to the log
        if found == True:
            inconsistency_log[dates[day]] = np.array(combinations)[consistency_index != 0]

        consistency_index = sum(consistency_index)/(len(consistency_index) - nans)

        consistency.append(consistency_index)

    return 1 - np.array(consistency), inconsistency_log


def transitivity(df):

    T = []
    nontransitive = []

    dates = np.unique(df['Date'])
    names = np.unique(df[['Winner', 'Loser']])

    for day in dates:
        record = []

        for name in names:
            wins = len(np.where(df[df['Date'] == day]['Winner'] == name)[0])

            record.append(wins)

        sorted_wins = sorted(record)
        sorted_wins = np.array(sorted_wins)
        transitive = sorted_wins == np.arange(0, len(record))

        if all(transitive):
            T.append(True)

        else:
            T.append(False)

            repeated_scores = sorted_wins[np.where(np.logical_not(transitive))[0]]
            repeated_scores = np.unique(repeated_scores)

            nontransitive_players = []
            for score in repeated_scores:

                nontransitive_players = np.append(nontransitive_players,
                                                  names[np.where(record == score)[0]])

            nontransitive.append(nontransitive_players)

    return np.array(T), nontransitive


def process_group_elo(path, score0=1000, k=100, accurate_dates=True, plotting=True):
    """
    Calculates the Elo ratings for a group of animals after each game

    Path = a path to an Excel (.xls) file containing the game results.
    File must contain at least 3 columns named "Date", "Winner", and "Loser" which specify the date of each game,
    name of the winner and loser of each game.

    score0 = initial score assigned to all animals

    k = scalar constant which specifies the maximum number of points transferred from the loser to the winner in one game

    accurate_dates = specifies whether to plot the Elo scores with accurate time intervals based on the testing dates.
    If not, intervals between testing points are assumed to be 1 day.

    plotting: boolean; specifies whether ranking plots should be generated
    """

    path = Path(path)

    # Loads the file
    try:
        df = pd.read_excel(path)

        # Finds the animal names and creates a rating and rank record
        names = np.unique(df[['Winner', 'Loser']])
        ratings = [np.array([score0])] * len(names)
        ratings = dict(zip(names, ratings))
        ranks = dict(zip(names, [[(len(names)+1)/2]]*len(names)))

        daily_ratings = ratings.copy()
        dates = np.unique(df['Date'])
        days = [0, 1]

        # Finds time differences between testing dates
        for n in range(len(dates) - 1):
            difference = (dates[n+1] - dates[n]).astype('timedelta64[D]').astype(int)
            days.append(days[-1] + difference)

        # Updates the rating and rank record for every game played in a day
        for date in dates:
            test_day = df[df['Date'] == date]

            # Updates ranking for each game
            for n in range(len(test_day)):

                game = test_day.iloc[n]

                daily_ratings[game['Winner']], daily_ratings[game['Loser']] = update_elo(daily_ratings[game['Winner']],
                                                                                         daily_ratings[game['Loser']],
                                                                                         1, k)

            # Concatenates the new daily ratings to the existing record
            ratings = concat_dicts(ratings, daily_ratings)

            # Converts the most recent ratings from dict to list
            end_of_day = np.array([])
            for animal in daily_ratings.items():
                end_of_day = np.append(end_of_day, [animal[1][-1]])

            # Ranks the players based on the most recent ratings and finds the best and worst player
            daily_ranks = rankdata(end_of_day)
            best = daily_ranks.argmax()
            best = names[best]
            worst = daily_ranks.argmin()
            worst = names[worst]
            daily_ranks = dict(zip(names, daily_ranks))
            ranks = concat_dicts(ranks, daily_ranks)

        print('Highest ranking animal is: ' + best)
        print('Lowest ranking animal is: ' + worst + '\n')

        S = stability_index(ranks)
        T, _ = transitivity(df)

        print('Hierarchy transitive over past 3 tests?')
        if all(T[-3:] == True):
            print('✓ \n')
        else:
            print('❌ \n')

        C, log = consistency(df)
        print('Inconsistent outcomes:')
        pprint(log)

        # Saves rankings to new excel file
        processed = pd.DataFrame(ranks)
        # processed.to_excel(path.parent.parent + '/Processed/' + path.name[:-4] + '_rankings.xls')

        if plotting:
            plt.rcParams['axes.spines.top'] = False
            plt.rcParams['axes.spines.right'] = False

            if not accurate_dates:
                days = np.arange(len(dates) + 1)

            # Plots the ranks
            plt.figure(figsize=[14, 11])
            plt.subplot(222)
            for animal in ranks.items():
                plt.plot(days, animal[1], label=animal[0], linewidth=2.5)

            plt.ylabel('Elo rank')
            plt.yticks(np.arange(len(names)) + 1)
            #plt.xticks(days)
            plt.xlabel('Session')

            # Plots the ratings
            plt.subplot(221)
            for animal in ratings.items():
                plt.plot(days, animal[1], label=animal[0], linewidth=2.5)

            plt.legend(ncol=len(names), loc=(0, 0.95), title='Mouse ID', frameon=False)
            plt.ylabel('Elo rating')
            plt.xlabel('Session')
            #plt.xticks(days)

            # Plots the consistency scores
            plt.subplot(223)
            plt.plot(days[2:], C)
            plt.ylim([0, 1.1])
            plt.ylabel('Consistency Index')
            plt.xlabel('Session')
            plt.xticks(days[2:])

            #Plots the stability index
            plt.subplot(224)
            plt.plot(days[2:], S)
            plt.xticks(days[2:])
            plt.ylim([0, 1.1])
            plt.ylabel('Hierarchy Stability Index')
            plt.xlabel('Session')

        return ratings, ranks, C, log, S, T

    except FileNotFoundError:
        warnings.warn("File not found - Are you connected to CAMP?")
