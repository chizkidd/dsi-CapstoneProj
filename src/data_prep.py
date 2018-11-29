import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean(csv_file):
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

    # create a new column - final game results differential (>, <, or = 0)
    noHT_epl_df['homeXResults'] = noHT_epl_df['homeGoalFT'] - noHT_epl_df['awayGoalFT']
    noHT_epl_df['awayXResults'] = noHT_epl_df['awayGoalFT'] - noHT_epl_df['homeGoalFT']

    # handle categories - categorical encoding
    # allow for classification modeling
    # a void [scikit-learn error: The least populated class in y has only 1 member]
    results_mask = {-8: -1, -7: -1, -6: -1, -5: -1, -4: -1, -3: -1, -2: -1, -1: -1,
                    0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    WDL_mask = {-8: 'L', -7: 'L', -6: 'L', -5: 'L', -4: 'L', -3: 'L', -2: 'L', -1: 'L', 0: 'D',
                1: 'W', 2: 'W', 3: 'W', 4: 'W', 5: 'W', 6: 'W', 7: 'W', 8: 'W'}

    noHT_epl_df['homeXResultsLabel'] = noHT_epl_df['homeXResults'].map(results_mask)
    noHT_epl_df['homeXResultsWDL'] = noHT_epl_df['homeXResults'].map(WDL_mask)
    noHT_epl_df['awayXResultsLabel'] = noHT_epl_df['awayXResults'].map(results_mask)
    noHT_epl_df['awayXResultsWDL'] = noHT_epl_df['awayXResults'].map(WDL_mask)

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

    noHT_epl_df['homeTeamCode'] = noHT_epl_df['homeTeam'].map(team_mask)
    noHT_epl_df['awayTeamCode'] = noHT_epl_df['awayTeam'].map(team_mask)
    noHT_epl_df['homeFormationCode'] = noHT_epl_df['homeFormation'].map(formation_mask).astype('category').cat.codes
    noHT_epl_df['awayFormationCode'] = noHT_epl_df['awayFormation'].map(formation_mask).astype('category').cat.codes
    return noHT_epl_df


def only_numerics_df(csv_file):
    '''

    :param csv_file: [type: csv file]
    :return: df: [type: pandas dataframe]

    A function that does feature engineering on a csv file and returns a dataframe for modeling
    '''

    cleaned_df = clean(csv_file)
    model_df = cleaned_df.drop(['awayTeamLineUp', 'homeTeamLineUp',
                               'awayManagerName', 'homeManagerName',
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
    '''
    cat_cols4 = ['refereeName', 'venueName']
    for column in cat_cols4:
        le = LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])
        encode[column] = le
        cat_map[column] = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(cat_map) #categories mapping
    return df


if __name__ == '__main__':
    csvFILEpath = input("Enter path to file that you wish to pre-process: (should be a .csv file) ")
    #df = clean(csvFILEpath)
    df2 = only_numerics_df(csvFILEpath)