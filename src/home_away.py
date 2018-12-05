import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_home_away(csv_file):
    '''
    :param csv_file: [type: csv file]
    :return: df: [type: pandas dataframe]

    A function that takes in a csv_file and reads in the file into a pandas dataframe
    then pre-processes the file via feature engineering for modeling readiness
    '''

    # read in the csv file to pandas dataframe
    df = pd.read_csv(csv_file)

    # focused on the english premier league (EPL) division
    epl_df = df[df.division == 'EPL']

    # drop halftime stats due to cumulative nature of fulltime stats
    noHT_epl_df = epl_df[epl_df.columns.drop(epl_df.filter(regex='HT', axis=1).columns)]

    # drop unnecessary extra index columns
    noHT_epl_df = noHT_epl_df.drop(['Unnamed: 0'], axis=1)

    # handle missing null values (column with max null % = 4.1%)
    noHT_epl_df = noHT_epl_df.fillna(noHT_epl_df.median())

    # feature engineering for:
    # attack and defence strength, dribbles_stopped ratio
    # shots on target and shots blocked percentage, and expected goals
    noHT_epl_df['homeDribbledPastFT'] = noHT_epl_df['homeDribbledPastFT'].replace(0, 0.1)
    noHT_epl_df['awayDribbledPastFT'] = noHT_epl_df['awayDribbledPastFT'].replace(0, 0.1)
    noHT_epl_df['homeAttackStrength'] = noHT_epl_df['homeGoalFT'] / noHT_epl_df['homeGoalFT'].mean()
    noHT_epl_df['homeDefenceStrength'] = noHT_epl_df['awayGoalFT'] / noHT_epl_df['awayGoalFT'].mean()
    noHT_epl_df['homeDribbleStopRatio'] = noHT_epl_df['homeDribblesWonFT'] / noHT_epl_df['homeDribbledPastFT']
    noHT_epl_df['homeSOTpercent'] = noHT_epl_df['homeShotsOnTargetFT'] / noHT_epl_df['homeShotsTotalFT']
    noHT_epl_df['homeShotsBlockedpercent'] = noHT_epl_df['homeShotsBlockedFT'] / noHT_epl_df['homeShotsTotalFT']

    noHT_epl_df['awayAttackStrength'] = noHT_epl_df['awayGoalFT'] / noHT_epl_df['awayGoalFT'].mean()
    noHT_epl_df['awayDefenceStrength'] = noHT_epl_df['homeGoalFT'] / noHT_epl_df['homeGoalFT'].mean()
    noHT_epl_df['awayDribbleStopRatio'] = noHT_epl_df['awayDribblesWonFT'] / noHT_epl_df['awayDribbledPastFT']
    noHT_epl_df['awaySOTpercent'] = noHT_epl_df['awayShotsOnTargetFT'] / noHT_epl_df['awayShotsTotalFT']
    noHT_epl_df['awayShotsBlockedpercent'] = noHT_epl_df['awayShotsBlockedFT'] / noHT_epl_df['awayShotsTotalFT']

    noHT_epl_df['homeXG'] = noHT_epl_df['homeAttackStrength'] * \
                            noHT_epl_df['awayDefenceStrength'] * noHT_epl_df['homeGoalFT'].mean()
    noHT_epl_df['awayXG'] = noHT_epl_df['awayAttackStrength'] * \
                            noHT_epl_df['homeDefenceStrength'] * noHT_epl_df['awayGoalFT'].mean()

    # home and away status
    # noHT_epl_df['homeStatus'] = 1
    # noHT_epl_df['awayStatus'] = 0

    # create a new column - final game results differential (>, <, or = 0)
    noHT_epl_df['homeXResults'] = noHT_epl_df['homeGoalFT'] - noHT_epl_df['awayGoalFT']
    noHT_epl_df['awayXResults'] = noHT_epl_df['awayGoalFT'] - noHT_epl_df['homeGoalFT']

    # handle categories - categorical encoding
    # allow for classification modeling [scikit-learn error: The least populated class in y has only 1 member]
    results_mask = {-8: -1, -7: -1, -6: -1, -5: -1, -4: -1, -3: -1, -2: -1, -1: -1,
                    0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    WDL_mask = {-8: 'L', -7: 'L', -6: 'L', -5: 'L', -4: 'L', -3: 'L', -2: 'L', -1: 'L', 0: 'D',
                1: 'W', 2: 'W', 3: 'W', 4: 'W', 5: 'W', 6: 'W', 7: 'W', 8: 'W'}

    noHT_epl_df['homeresultsLabel'] = noHT_epl_df['homeXResults'].map(results_mask)
    # noHT_epl_df['homeXResultsWDL'] = noHT_epl_df['homeXResults'].map(WDL_mask)
    noHT_epl_df['awayresultsLabel'] = noHT_epl_df['awayXResults'].map(results_mask)
    # noHT_epl_df['awayXResultsWDL'] = noHT_epl_df['awayXResults'].map(WDL_mask)

    # encoding categories for modeling
    # encoding team-names, team formation
    team_mask = {'Arsenal': 1, 'Swansea': 2, 'West Brom': 3, 'Newcastle': 4, 'Liverpool': 5,
                 'West Ham': 6, 'Chelsea': 7, 'Everton': 8, 'Man Utd': 9, 'Southampton': 10,
                 'Stoke': 11, 'Sunderland': 12, 'Wigan': 13, 'Tottenham': 14, 'Fulham': 15, 'Reading': 16,
                 'Norwich': 17, 'QPR': 18, 'Man City': 19, 'Aston Villa': 20, 'Crystal Palace': 21,
                 'Cardiff': 22, 'Hull': 23, 'Burnley': 24, 'Leicester': 25, 'Watford': 26,
                 'Bournemouth': 27, 'Middlesbrough': 28}
    # inv_team_mask = {v: k for k, v in team_mask.items()}

    # http://www.freeyouthsoccerdrills.com/soccer-formations.html
    # https://bleacherreport.com/articles/1375589-15-tactical-formations-and-what-theyre-good-for#slide7
    # https://www.dummies.com/sports/soccer/choosing-a-formation-in-soccer/
    '''
    AA(6) - highly attacking - '343', '433','352', '3412', '3142', '4240' 
    AB(5) - attacking - '3421', '4141','3511', '4132', '4312'
    B(4) - balanced - '4231', '442', 41212', '4222',
    BD(3) - defensive -  '451', '4411', '4321'
    DD(3) - highly defensive - '541', '532', '343d'
    '''
    formation_mask = {'343': 'AA', '4231': 'B', '451': 'BD', '4411': 'BD', '442': 'B',
                      '433': 'AA', '3421': 'AB', '4141': 'AB', '352': 'AA', '3511': 'AB',
                      '3412': 'AA', '41212': 'B', '541': 'DD', '532': 'DD', '4222': 'B',
                      '4321': 'BD', '4132': 'AB', '3142': 'AA', '4312': 'AB', '4240': 'AA',
                      '343d': 'DD'}

    # noHT_epl_df['homeTeamCode'] = noHT_epl_df['homeTeam'].map(team_mask)
    # noHT_epl_df['awayTeamCode'] = noHT_epl_df['awayTeam'].map(team_mask)
    noHT_epl_df['homeFormationCode'] = noHT_epl_df['homeFormation'].map(formation_mask).astype('category').cat.codes
    noHT_epl_df['awayFormationCode'] = noHT_epl_df['awayFormation'].map(formation_mask).astype('category').cat.codes
    return noHT_epl_df


def get_model_df(csv_file):
    '''

    :param csv_file: [type: csv file]
    :return: df: [type: pandas dataframe]

    A function that does feature engineering on a csv file, drops some features 
    and  returns a dataframe for modeling
    '''

    cleaned_df = create_home_away(csv_file)
    model_df = cleaned_df.drop(['date', 'awayTeamLineUp', 'homeTeamLineUp',
                                'awayFormation', 'homeFormation',
                               'awayManagerName', 'homeManagerName',
                                'awayDribblesWonFT', 'homeDribblesWonFT',
                                'awayDribbledPastFT', 'homeDribbledPastFT',
                                'awayGoalFT', 'homeGoalFT',
                                'awayShotsOnTargetFT', 'homeShotsOnTargetFT',
                                'awayShotsBlockedFT','homeShotsBlockedFT',
                                'awayShotsTotalFT', 'homeShotsTotalFT',
                                'awayTacklesTotalFT', 'homeTacklesTotalFT',
                                'awayCornersTotalFT', 'homeCornersTotalFT',
                                'awayXResults', 'homeXResults',
                                'refereeName', 'venueName',
                                'division'], axis=1)
    model_df = category_encoder(model_df)
    sorted_col_model_df = model_df.sort_index(axis=1)
    return sorted_col_model_df


def category_encoder(df):
    '''
    :param df: dataframe
    :return: df: dataframe

    Categorical label encoding for ['refereeName', 'venueName']

    '''
    encode = {}
    cat_map = {}
    '''
    cat_cols = ['homeFormation', 'refereeName','awayManagerName','awayTeam', 
    'awayFormation', 'homeTeam','homeManagerName','venueName']
    cat_cols2 = ['refereeName', 'awayManagerName', 'homeManagerName', 'venueName']
    cat_cols3 = ['refereeName', 'venueName','homeFormation','awayFormation']
    cat_cols4 = ['refereeName', 'venueName']
    '''
    cat_cols4 = [ 'homeTeam', 'awayTeam']
    for column in cat_cols4:
        le = LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])
        encode[column] = le
        cat_map[column] = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(cat_map) #categories mapping
    return df


def join_home_away(csv_file):
    '''
    :param csv_file:  [type: csv file]
    :return: df: {[ype: pandas dataframe}

    A function that concatenates home and away data and returns a dataframe 
    for home_away, home, and away respectively

    '''
    # ready_df = data_prep.only_numerics_df('../data/FootballEurope/FootballEurope.csv')
    # ensure right dataframe input format
    ready_df = get_model_df(csv_file)
    if 'resultsLabel' in ready_df.columns:
        ready_df = ready_df.drop(['resultsLabel'], axis=1)
    data = ready_df
    data = data[sorted(data.columns)]
    # away_cols = data.columns[:21]
    # home_cols = data.columns[21:42]
    away_cols = data.filter(regex='away', axis=1).columns
    home_cols = data.filter(regex='home', axis=1).columns
    data_home = data.ix[:, home_cols]
    data_away = data.ix[:, away_cols]
    new_col = []
    new_cols = []
    for col in data_home.columns:
        new_col.append(col.split('home')[1:])
    # print(new_col)

    for row in new_col:
        new_cols.append(row[0])
    # print(new_cols)

    # home_away_col = [item for sublist in new_col for item in sublist]
    # print()
    # print(home_away_col)

    data_away.columns = new_cols
    data_home.columns = new_cols

    data_home['game_id'] = ready_df['id']
    data_away['game_id'] = ready_df['id']

    data_home['at_home'] = 1
    data_away['at_home'] = 0

    # data_home['date'] = ready_df['date']
    # data_away['date'] = ready_df['date']

    clust_data = data_home.append(data_away, ignore_index=True)
    return clust_data, data_home, data_away


if __name__ == '__main__':
    csvFILEpath = input("Enter path to file that you wish to pre-process: (should be a .csv file) ")
    #df = clean(csvFILEpath)
    df2, df_hom, df_awy = join_home_away(csvFILEpath)
