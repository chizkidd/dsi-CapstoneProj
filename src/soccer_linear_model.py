from rolling_means import final_df_rolling_means
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize


def get_model_df(csv_file,num_games=10, min_games_reqd=5):
    '''
    :param csv_file:
    :return:

    A function to get perform rolling averages

    '''

    roll_df = final_df_rolling_means(csv_file, num_games, min_games_reqd)
    roll_mat = roll_df.as_matrix()
    roll_mat_list = []
    for i in range(int(len(roll_mat) / 2)):
        roll_mat_list.append(list(roll_mat[2 * i]) + (list(roll_mat[2 * i + 1])))
    # print(roll_mat_list)
    roll_cols = ('Home_' + roll_df.columns).append('Away_' + roll_df.columns)
    # print(len(roll_cols))
    final_roll_df = pd.DataFrame(roll_mat_list, columns=roll_cols, index=roll_df.index.get_level_values('game_id')[::2])
    final_roll_df2 = final_roll_df.drop(['Home_TeamCode', 'Away_TeamCode', 'Home_at_home', 'Away_at_home',
                                         'Away_Total_GoalsFT', 'Home_XResults', 'Away_XResults',
                                         'Home_RatingsFT', 'Away_RatingsFT'], axis=1)
    model_roll_df = final_roll_df2.dropna()

    return model_roll_df


def linear_model(model_df, Linear_model, test_size, scaled=None):
    '''
    A function that trains a linear model and retrieves metrics scores on test data

    input: model_df: dataframe to use for modeling
           Linear_model: Linear model (linear regression, Lasso, Ridge)
           test_size: train_test size split fraction WRT to test data (between 0 and 1)
           input_train_test_data: default-->None, if [no_train_test_split] = True, give input test and train data
           no_train_test_split: default-->False, but if True, uses input test and train data set
           scaled: default-->False, but if True specify what scaling form ('Standard', 'MinMax', 'Normalized')
    output: model, r-squared, Mean Absolute error, Root Mean Square Error, dataframe with actual and predicted values
    '''
    if scaled == 'Standard':
        scaler = StandardScaler().fit(model_df)
        model_scaled_df = scaler.transform(model_df)
        model_df = pd.DataFrame(model_scaled_df, columns=model_df.columns,
                                index=model_df.index.get_level_values('game_id'))
    elif scaled == 'MinMax':
        min_max_scaler = MinMaxScaler().fit(model_df)
        model_scaled_df = min_max_scaler.transform(model_df)
        model_df = pd.DataFrame(model_scaled_df, columns=model_df.columns,
                                index=model_df.index.get_level_values('game_id'))
    elif scaled == 'Normalize':
        model_df = pd.DataFrame(normalize(model_df), columns=model_df.columns,
                                index=model_df.index.get_level_values('game_id'))

    else:
        model_df = model_df

    # input_train_test_data = None, no_train_test_split=False
    #if no_train_test_split:
    #    X_train, X_test, y_train, y_test = input_train_test_data
    #else:
    #    X_train, X_test, y_train, y_test = train_test_split(model_df.drop('Home_Total_GoalsFT', axis=1),
    #                                                    model_df['Home_Total_GoalsFT'], test_size=test_size,
    #                                                    random_state=789)

    X_train, X_test, y_train, y_test = train_test_split(model_df.drop('Home_Total_GoalsFT', axis=1),
                                                        model_df['Home_Total_GoalsFT'], test_size=test_size,
                                                        random_state=789)

    Linear_model.fit(X_train, y_train)
    rsq = abs(Linear_model.score(X_test, y_test))
    preds = Linear_model.predict(X_test)
    results_df = pd.DataFrame(y_test.values, columns=['actual'], index=y_test.index.get_level_values('game_id'))
    results_df['predictions'] = preds
    results_df['differences'] = results_df['actual'] - results_df['predictions']
    MAE = sum(np.abs(y_test - preds)) / len(preds)
    RootMSE = np.sqrt(np.sum(((y_test - preds) ** 2)) / (len(preds) - 1))
    print('Model:{}'.format(Linear_model))
    print('R-squared:{}'.format(rsq))
    print('Mean Abs Error:{}'.format(MAE))
    print('RMSE:{}'.format(RootMSE))
    return Linear_model, rsq, MAE, RootMSE, results_df


if __name__ == '__main__':
    csvFILEpath = input("Enter path to file that you wish to pre-process: (should be a .csv file) ")
    df = get_model_df(csvFILEpath)
    lassoL1 = Lasso(alpha=0.005, tol=.0002)
    ridgeL2 = Ridge(alpha=0.005)
    linreg = LinearRegression()
    LassoPred = linear_model(df, lassoL1, 0.25, 'MinMax')
    LinRegPred = linear_model(df, linreg, 0.25, 'MinMax')
    RidgePred = linear_model(df, ridgeL2, 0.25, 'MinMax')