import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from preprocessor import feat_eng, create_model_df, category_encoder


def create_league_table(csv_file):
    '''
    :param csv_file: input a csv file for pre-processing [type: csv file]
    :return: df: cumulative table [type: pandas dataframe]

    Recreate a cumulative League Table over the seasons of data collection.
    Show teams and corresponding points accumulated using specific league point system.
    '''
    premtable = feat_eng(csv_file)

    # encoding for results (win, loss, draw) for home team and away team
    EPL_res_keyH = {'W': 3, 'L': 0, 'D': 1}
    EPL_res_keyA = {'W': 0, 'L': 3, 'D': 1}

    # results WRT to the hometeam and awayteam
    premtable['EPLresultH'] = premtable.resultsWDL.map(EPL_res_keyH)
    premtable['EPLresultA'] = premtable.resultsWDL.map(EPL_res_keyA)

    # encode team name back from numerical code to actual team name
    team_mask = {'Arsenal': 1, 'Swansea': 2, 'West Brom': 3, 'Newcastle': 4, 'Liverpool': 5,
                 'West Ham': 6, 'Chelsea': 7, 'Everton': 8, 'Man Utd': 9, 'Southampton': 10,
                 'Stoke': 11, 'Sunderland': 12, 'Wigan': 13, 'Tottenham': 14, 'Fulham': 15, 'Reading': 16,
                 'Norwich': 17, 'QPR': 18, 'Man City': 19, 'Aston Villa': 20, 'Crystal Palace': 21,
                 'Cardiff': 22, 'Hull': 23, 'Burnley': 24, 'Leicester': 25, 'Watford': 26,
                 'Bournemouth': 27, 'Middlesbrough': 28}
    inv_team_mask = {v: k for k, v in team_mask.items()}
    premtable['EPLhomeTeam'] = premtable['homeTeam'].map(inv_team_mask)
    premtable['EPLawayTeam'] = premtable['awayTeam'].map(inv_team_mask)

    HomePtTally_df = premtable.groupby('EPLhomeTeam')[
        'EPLhomeTeam', 'EPLawayTeam', 'EPLresultH'].sum().sort_values('EPLresultH', ascending=False)
    AwayPtTally_df = premtable.groupby('EPLawayTeam')[
        'EPLhomeTeam', 'EPLawayTeam', 'EPLresultA'].sum().sort_values('EPLresultA', ascending=False)

    epltable = pd.concat([HomePtTally_df, AwayPtTally_df], axis=1)
    epltable['EPLresult'] = epltable['EPLresultH'] + epltable['EPLresultA']
    # print(epltable.sort_values('EPLresult', ascending=False))
    return epltable


def plot_league_table(csv_file):
    '''

    :param csv_file: input a csv file [type: csv file]
    :return: plot of league table [type: matplotLib object]

    Make a visual plot of the cumulative table over the seasons of data collection using specific league
    point system. The cumulative league table visualization is shown here through matplotLib.

    '''
    cum_league_table = create_league_table(csv_file)
    cum_league_table['EPLresult'].sort_values(
        ascending=False).plot(kind='bar', figsize=(12, 8), label='EPL points system - Bar')
    plt.plot(cum_league_table['EPLresult'].sort_values(ascending=False).index,
             cum_league_table['EPLresult'].sort_values(ascending=False).values,
             linestyle='-', marker='o', label="EPL points system - Line", color='orange')
    plt.title('Point Tally per Team from 2012/13 to 2016/2017', size=20, fontweight='bold')
    plt.legend(loc='upper right', fontsize=13)
    plt.xlabel('EPL team', size=15)
    plt.ylabel('Actual Cumulative Point Tally', size=15)
    plt.show()


if __name__ == '__main__':
    csv_file = input("Enter path to file that you wish to preprocess and plot: (should be a .csv file) ")
    plot_table = plot_league_table(csv_file)