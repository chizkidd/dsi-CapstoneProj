import matplotlib.pyplot as plt
from data_prep import clean, only_numerics_df


def clean_data(csv_file):
    '''
    :param csv_file: [type: csv file]
    :return: df: [type: pandas dataframe]

    A function that preps data for rolling means / modelling

    '''
    df_clean = only_numerics_df(csv_file)
    return df_clean


def dominancy_at_home(csv_file):
    '''
    :param csv_file:
    :return: histogram plot

    A function that plots team goal differential when the team plays at home

    '''
    df2 = clean_data(csv_file)
    home_df2 = df2[df2.filter(regex='home', axis=1).columns]
    plt.plot(home_df2.groupby(['homeTeam'])['homeXResults', 'homeXResultsLabel'].mean().index,
             home_df2.groupby(['homeTeam'])['homeXResults', 'homeXResultsLabel'].mean().values[:, 0],
             linestyle='-', marker='o', label="EPL points system", color='orange')
    plt.plot(home_df2.groupby(['homeTeam'])['homeXResults', 'homeXResultsLabel'].mean().index,
             home_df2.groupby(['homeTeam'])['homeXResults', 'homeXResultsLabel'].mean().values[:, 1],
             linestyle='-.', marker='o', label="Label system [1,0,-1]", color='green')
    home_df2.groupby(['homeTeam'])['homeXResults'].mean().plot(kind='bar', figsize=(12, 8), label='EPL points system')
    plt.legend(loc='upper right', fontsize=13)
    plt.xlabel("Teams", size=15)
    plt.ylabel("Home score - Away score", size=15)
    plt.title('Average Home Goal Differential By Home Team', size=20, fontweight='bold')
    plt.show()


def dominancy_when_away(csv_file):
    '''
    :param csv_file:
    :return: histogram plot

    A function that plots team goal differential when the team plays as away team

    '''
    df2 = clean_data(csv_file)
    away_df2 = df2[df2.filter(regex='away', axis=1).columns]
    plt.plot(away_df2.groupby(['awayTeam'])['awayXResults', 'awayXResultsLabel'].mean().index,
             away_df2.groupby(['awayTeam'])['awayXResults', 'awayXResultsLabel'].mean().values[:, 0],
             linestyle='-', marker='o', label="EPL points system", color='orange')
    plt.plot(away_df2.groupby(['awayTeam'])['awayXResults', 'awayXResultsLabel'].mean().index,
             away_df2.groupby(['awayTeam'])['awayXResults', 'awayXResultsLabel'].mean().values[:, 1],
             linestyle='-.', marker='o', label="Label system [1,0,-1]", color='green')
    away_df2.groupby(['awayTeam'])['awayXResults'].mean().plot(kind='bar', figsize=(12, 8), label='EPL points system')
    plt.legend(loc='lower right', fontsize=13)
    plt.xlabel("Teams", size=15)
    plt.ylabel("Away score - Home score", size=15)
    plt.title('Average Away Goal Differential By Away Team', size=20, fontweight='bold')
    plt.show()


if __name__ == '__main__':
    csvFILEpath = input("Enter path to file for plotting: (should be a .csv file) -->"
                        "../data/FootballEurope/FootballEurope.csv ")
    dominancy_at_home(csvFILEpath)
    dominancy_when_away(csvFILEpath)
