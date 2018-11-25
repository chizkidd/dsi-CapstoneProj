import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from preprocessor import feat_eng, category_encoder


def create_league_table(csv_file):
    '''
    :param df: dataframe
    :return: df: dataframe

    Recreate a cummulative League Table over the seasons of data collection
    Show teams and corresponding points accumulated using league point system
    '''
    premtable = feat_eng(csv_file)

    # results W.R.T. to the hometeam and awayteam
    premtable['EPLresultH'] = premtable.resultsWDL.map(EPL_res_keyH)
    premtable['EPLresultA'] = premtable.resultsWDL.map(EPL_res_keyA)

    # encode team
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
        'EPLhomeTeam', 'EPLawayTeam', 'results', 'EPLresultH'].sum().sort_values('EPLresultH', ascending=False)
    AwayPtTally_df = premtable.groupby('EPLawayTeam')[
        'EPLhomeTeam', 'EPLawayTeam', 'results', 'EPLresultA'].sum().sort_values('EPLresultA', ascending=False)

    epltable = pd.concat([HomePtTally_df, AwayPtTally_df], axis=1)
    epltable['EPLresult'] = epltable['EPLresultH'] + epltable['EPLresultA']
    epltable = epltable.sort_values('EPLresult', ascending=False)
    return epltable


def plot_league_table(csv_file):
    '''

    :param df:
    :return: plot of league table

    make a visual plot of the league table created in 'league_table'

    '''
    cum_league_table = create_league_table(csv_file)
    cum_league_table.plot(kind='bar', figsize=(10, 5))
    plt.title('Point Tally per Team from 2012/13 to 2016/2017')
    plt.xlabel('EPL team')
    plt.ylabel('Actual Cumulative Point Tally')
    plt.show()


if __name__ == '__main__':
    csv_file = Input("Enter path to file that you wish to preprocess and plot: (should be a .csv file) ")
    plot_table = plot_league_table(csv_file)