from preprocessor import feat_eng, category_encoder, create_model_df
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize


def cluster_data_prep(csv_file):
    '''
    A function that concatenates home and away data stats

    :param csv_file:  [type: csv file]
    :return: (df, df, df): [type: a tuple of pandas dataframe]
    '''
    ready_df = create_model_df(csv_file)  # ensure right dataframe input format
    data = ready_df.drop(['resultsLabel'], axis=1)
    data = data[sorted(data.columns)]
    away_cols = data.columns[:21]
    home_cols = data.columns[21:42]
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

    clust_data = data_home.append(data_away, ignore_index=True)
    return clust_data, data_home, data_away


def trend(csv_file, n_clusters=20, scaled=False):
    '''
    A function to run data_prep, k-means clustering, and tactical identification.
    (Also influences feature selection)

    :param csv_file: [type: csv file]
    :param n_clusters: [type: int]
    :param scaled: default--> False ; normalize data-set using MinMax scaling
    :return: df: [type: pandas dataframe]  tactics pattern cluster dataframe
    '''
    clust_data, _, _ = cluster_data_prep(csv_file)
    tactics_df = tactical_clust_patterns(clust_data, n_clusters, scaled)
    return tactics_df


def predict(clust_data, n_clusters=20, scaled=False):
    '''
    A function that returns predictions based on k-means clustering

    :param clust_data: [type: pandas dataframe]
    :param n_clusters: [type: int] - number of clusters for clustering
    :param scaled: default--> False ; normalize data-set using MinMax scaling
    :return: dataframe: [type: pandas dataframe] - tactical patterns of each cluster
    '''
    if scaled:
        minmax = MinMaxScaler().fit(clust_data)
        clust_data_minmax_scaled = minmax.transform(clust_data)
        clust_data_scaled = pd.DataFrame(clust_data_minmax_scaled, columns=clust_data.columns)
        clustData = clust_data_scaled
    else:
        clustData = clust_data
    kmeans = KMeans(n_clusters, random_state=123)
    kmeans.fit(clustData.values)
    y_pred = kmeans.predict(clustData.values)
    return y_pred


def tactical_clust_patterns(clust_data, n_clusters=20, scaled=False):
    '''
    A function that returns a cluster dataframe showing different tactical similarities in each cluster
    The cluster dataframe is computed using k-means clustering (cluster_centers)

    :param clust_data: [type: pandas dataframe]
    :param n_clusters: [type: int] - number of clusters for clustering
    :param scaled: default--> False ; normalize data-set using MinMax scaling
    :return: dataframe: [type: pandas dataframe] - tactical patterns of each cluster
    '''
    if scaled:
        minmax = MinMaxScaler().fit(clust_data)
        clust_data_minmax_scaled = minmax.transform(clust_data)
        clust_data_scaled = pd.DataFrame(clust_data_minmax_scaled, columns=clust_data.columns)
        clustData = clust_data_scaled
    else:
        clustData = clust_data
    kmeans = KMeans(n_clusters, random_state=123)
    kmeans.fit(clustData.values)
    clust_kmeans = pd.DataFrame(kmeans.cluster_centers_, columns=clustData.columns)
    clust_stats = clustData.describe(percentiles=[0.1, 0.35, 0.65, 0.9])
    clust_kmeans_exp = clust_kmeans.copy()
    for col in clust_kmeans.columns:
        for i, row in enumerate(clust_kmeans[col]):
            # print(i, row)
            # print(i, col, clust_kmeans.ix[i,col])
            if clust_kmeans.ix[i, col] < clust_stats.ix['10%', col]:
                clust_kmeans_exp.ix[i, col] = 'V. Low'
            elif clust_kmeans.ix[i, col] < clust_stats.ix['35%', col]:
                clust_kmeans_exp.ix[i, col] = 'Low'
            elif clust_kmeans.ix[i, col] > clust_stats.ix['90%', col]:
                clust_kmeans_exp.ix[i, col] = 'V. High'
            elif clust_kmeans.ix[i, col] > clust_stats.ix['65%', col]:
                clust_kmeans_exp.ix[i, col] = 'High'
            else:
                clust_kmeans_exp.ix[i, col] = 'Medium'
    clust_kmeans_exp = clust_kmeans_exp.T
    if n_clusters == 20:
        soccer_tactics = {0: '', 1: '', 2: '', 3: '', 4: '', 5: '', 6: '', 7: '', 8: '', 9: '',
              10: '', 11: '', 12:'', 13 :'', 14 :'', 15: '', 16 :'', 17 :'', 18 :'', 19: ''}
        clust_kmeans_exp.ix['Tactic', :] = soccer_tactics
    else:
        clust_kmeans_exp = clust_kmeans_exp
    return clust_kmeans_exp


def percentiles_clusters(clust_kmeans_exp):
    '''
    A function that breaks the tactics cluster dataframe into 5 different dataframes
    (Very High, High, Medium, Low, Very Low)

    :param clust_kmeans_exp: [type - pandas dataframe] - tactical patterns of each cluster
    :return: (df, df, df, df, df): [type: tuple of pandas dataframe]
    '''
    clust_very_low = clust_kmeans_exp[clust_kmeans_exp == 'V. Low'].fillna('-')
    clust_low = clust_kmeans_exp[clust_kmeans_exp == 'Low'].fillna('-')
    clust_very_high = clust_kmeans_exp[clust_kmeans_exp == 'V. High'].fillna('-')
    clust_high = clust_kmeans_exp[clust_kmeans_exp == 'High'].fillna('-')
    clust_med = clust_kmeans_exp[clust_kmeans_exp == 'Medium'].fillna('-')
    return clust_very_high, clust_high, clust_med, clust_low, clust_very_low


def high_percentiles(clust_kmeans_exp):
    '''
    A function that breaks the tactics cluster dataframe into 5 different dataframes
    (Very High, High, Medium, Low, Very Low) and then returns (Very High, High) in a
    dataframe

    :param clust_kmeans_exp: type: pandas dataframe] - tactical patterns of each cluster
    :return:
    '''
    high_percents = clust_kmeans_exp[(clust_kmeans_exp[clust_kmeans_exp.columns] == 'V. High') |
                     (clust_kmeans_exp[clust_kmeans_exp.columns] == 'High')].fillna('-')
    return high_percents


def low_percentiles(clust_kmeans_exp):
    '''
    A function that breaks the tactics cluster dataframe into 5 different dataframes
    (Very High, High, Medium, Low, Very Low) and then returns (Very Low, Low) in a
    dataframe

    :param clust_kmeans_exp: type: pandas dataframe] - tactical patterns of each cluster
    :return:
    '''
    low_percents = clust_kmeans_exp[(clust_kmeans_exp[clust_kmeans_exp.columns] == 'V. Low') |
                     (clust_kmeans_exp[clust_kmeans_exp.columns] == 'Low')].fillna('-')
    return low_percents


def k_elbow_plot(csv_file, scaled=False):
    '''
    A function that creates a k-elbow plot to help identify optimal k value

    :param csv_file: [type: csv]
    :param scaled: default--> False ; normalize data-set using MinMax scaling
    :return:
    '''
    # k means determine k
    clust_data, _, _ = cluster_data_prep(csv_file)
    if scaled:
        minmax = MinMaxScaler().fit(clust_data)
        clust_data_minmax_scaled = minmax.transform(clust_data)
        clust_data_scaled = pd.DataFrame(clust_data_minmax_scaled, columns=clust_data.columns)
        clustData = clust_data_scaled
    else:
        clustData = clust_data
    distortions = []
    cluster_range = range(1, 30)
    for k in cluster_range:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(clustData)
        distortions.append(sum(np.min(cdist(clustData,
                        kmeanModel.cluster_centers_, 'euclidean'), axis=1))/clustData.shape[0])
    # print(distortions)

    # Plot the elbow
    plt.grid(b=True, which='major', linestyle='-')
    plt.plot(cluster_range, distortions, 'bx-')
    plt.xlabel('k - number of clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.xticks(np.arange(0, max(K)+1, 2.0))
    plt.show()


if __name__ == '__main__':
    csvFILEpath = input("Enter path to file that you wish to pre-process: (should be a .csv file) ")
    kmeans_tactics_df = trend(csvFILEpath, 20, True)
    low_and_very_low = low_percentiles(kmeans_tactics_df)
    high_and_very_high = high_percentiles(kmeans_tactics_df)
    k_elbow_plot(csvFILEpath)

