from data_prep import clean, only_numerics_df


def clean_data(csv_file):
    '''
    A function that preps data for rolling means / modelling

    :param df: raw dataframe
    :return: df: cleaned dataframe
    '''
    df_clean = only_numerics_df(csv_file)
    return df_clean


def cluster_data_prep(ready_df):
    '''
    A function that concatenates home and away data

    :param: df:  {type: pandas dataframe]
    :return: daf: {type: pandas dataframe}
    '''
    # ready_df = data_prep.only_numerics_df('../data/FootballEurope/FootballEurope.csv')
    # ensure right dataframe input format
    if 'resultsLabel' in ready_df.columns:
        ready_df = ready_df.drop(['resultsLabel'], axis=1)
    data = ready_df
    data = data[sorted(data.columns)]
    away_cols = data.columns[:27]
    home_cols = data.columns[28:55]
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

    home_away_col = [item for sublist in new_col for item in sublist]
    # print()
    # print(home_away_col)

    data_away.columns = new_cols
    data_home.columns = new_cols

    data_home['game_id'] = ready_df['id']
    data_away['game_id'] = ready_df['id']

    data_home['at_home'] = 1
    data_away['at_home'] = 0

    data_home['date'] = ready_df['date']
    data_away['date'] = ready_df['date']

    clust_data = data_home.append(data_away, ignore_index=True)
    return clust_data, data_home, data_away


if __name__ == '__main__':
    csvFILEpath = input("Enter path to file that you wish to pre-process: (should be a .csv file) ")
    #df = clean(csvFILEpath)
    df = clean(csvFILEpath)
    df2 = cluster_data_prep(df)