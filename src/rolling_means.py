import pandas as pd
from dateutil.parser import parse
from data_prep import clean, only_numerics_df


def clean_data(csv_file):
    '''
    :param csv_file: [type: csv file]
    :return: df: [type: a tuple of pandas dataframe]

    A function that preps data for rolling means / modelling

    '''
    df_clean = only_numerics_df(csv_file)
    return df_clean


def cluster_data_prep(csv_file):
    '''
    :param csv_file:  [type: csv file]
    :return: (df, df, df): [type: pandas dataframe]

    A function that concatenates home and away data

    '''
    # ready_df = data_prep.only_numerics_df('../data/FootballEurope/FootballEurope.csv')
    # ensure right dataframe input format
    ready_df = clean_data(csv_file)
    if 'resultsLabel' in ready_df.columns:
        ready_df = ready_df.drop(['resultsLabel'], axis=1)
    data = ready_df
    data = data[sorted(data.columns)]
    # away_cols = data.columns[:27]
    # home_cols = data.columns[28:55]
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

    data_home['date'] = ready_df['date']
    data_away['date'] = ready_df['date']

    clust_data = data_home.append(data_away, ignore_index=True)
    return clust_data, data_home, data_away


def total_goals_per_game(clust_df):
    '''
    :param clust_df:  [type: pandas dataframe]
    :return: df:  [type: pandas dataframe]

    A function that returns a dataframe containing the total number of goals scored in a game

    '''
    total_stats_per_game = pd.DataFrame(clust_df.groupby('game_id').sum()['GoalFT'])
    total_stats_per_game['Total_GoalsFT'] = total_stats_per_game['GoalFT']
    total_stats_per_game = total_stats_per_game.drop('GoalFT',axis=1)
    total_stats_per_game.head()
    clust_df = clust_df.join(total_stats_per_game, on='game_id')
    return clust_df


def create_days_opener(df):
    '''
    :param  df: dataframe containing home and away joined under the same stats
    :return: dataframe containing days after opener

    A function that adds 'a days after opener' column to an input dataframe

    '''
    df['date'] = pd.Series([parse(df['date'][i]) for i in range(len(df))])
    df['days_after_opener'] = df['date'] - df['date'].min()
    df = df.drop('date', axis=1)
    return df


def single_team_rolling_means(team_df, num_games, min_games_reqd):
    '''
    :param  team_df: team dataframe
    :param num_games
    :param min_games_reqd
    :return: df: dataframe of shifted rolling mean

    A function that calculates the rolling means per team and returns a dataframe output

    '''
    roll = team_df.rolling(num_games, min_games_reqd).mean()
    roll.drop(['GoalFT', 'XResults', 'XResultsLabel'], axis=1, inplace=True)
    return roll.shift(1)


def teams_dict_rolling_means(df, num_games, min_games_reqd):
    '''
    :param  df: dataframe containing home and away joined under the same stats and also with
    days after opener added and (DATE DROPPED)
    :param num_games
    :param min_games_reqd
    :return: {}: dictionary of rolling average of each team using a certain number of games and
    a minimum number of games

    A function that creates a rolling average team stats dataframe from a given dataframe

    '''
    # away team aerials stats exactly the same as home team aerial stats for >99% of total data (all 5 leagues)
    # df = df.drop('AerialsTotalFT', axis=1)
    games = df.groupby(['days_after_opener', 'game_id', 'Team']).sum()
    rolling_means_dict = {}
    for team in games.index.get_level_values('Team'):
        team_df = games[games.index.get_level_values('Team') == team]
        rolling_means_dict[team] = single_team_rolling_means(team_df, num_games, min_games_reqd)
    return rolling_means_dict


def teams_df_rolling_means(roll_means_dict):
    '''
    :param  roll_means_dict: dictionary of all teams rolling means
    :return: dataframe of all teams rolling means

    A function that creates a rolling means dataframe from a rolling means dictionary

    '''
    frames = []
    for team in roll_means_dict:
        frames.append(roll_means_dict[team])
    roll_means_df = pd.concat(frames)

    days_after_opener_idx = roll_means_df.index.get_level_values('days_after_opener')
    game_id_idx = roll_means_df.index.get_level_values('game_id')
    team_idx = roll_means_df.index.get_level_values('Team')
    rollin_means_df = roll_means_df.groupby([days_after_opener_idx, game_id_idx, team_idx]).sum()
    return rollin_means_df


def final_df_rolling_means(csv_file, num_games=10, min_games_reqd=5):
    '''
    :param csv_file
    :param num_games
    :param min_games_reqd
    :return: dataframe:

    A function that performs rolling means on a processed csv_file and returns a dataframe apt for modelling, and 
    re-indexes the dataframe with 3 indices (days_after_opener, game_id, team)

    '''

    # implement the functions above
    df_home_away, df_home, df_away = cluster_data_prep(csv_file)
    df_ha_stats = total_goals_per_game(df_home_away)
    days_df = create_days_opener(df_ha_stats)
    roll_dict = teams_dict_rolling_means(days_df,num_games, min_games_reqd)
    roll_df = teams_df_rolling_means(roll_dict)

    # re-index and group dataframe with 3 indices (days_after_opener, game_id, team) in the proper order
    days_after_opener_idx = roll_df.index.get_level_values('days_after_opener')
    game_id_idx = roll_df.index.get_level_values('game_id')
    team_idx = roll_df.index.get_level_values('Team')
    roll_grouped_df = roll_df.groupby([days_after_opener_idx, game_id_idx, team_idx]).sum()
    final_results_df = days_df[['game_id', 'Team', 'GoalFT', 'XResults', 'days_after_opener']]
    game_results_df = final_results_df.groupby(['days_after_opener', 'game_id', 'Team']).mean()

    # join total_game_stats dataframe to grouped dataframe
    roll_means_df = roll_grouped_df.join(game_results_df)
    return roll_means_df


if __name__ == '__main__':
    csvFILEpath = input("Enter path to file that you wish to pre-process: (should be a .csv file) ")
    roll_df = final_df_rolling_means(csvFILEpath)

