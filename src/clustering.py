from preprocessor import feat_eng, category_encoder, create_model_df
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def trend(csv_file, n_clusters=20):
    '''
    A function to run data_prep, k-means clustering, and tactical identification.
    (Also influences feature selection)

    :param csv_file: [type: csv file]
    :param n_clusters: [type: int]
    :return: df: [type: pandas dataframe]  tactics pattern cluster dataframe
    '''
    clust_data, _, _ = cluster_data_prep(csv_file)
    tactics_df = tactical_clust_patterns(clust_data, n_clusters)
    return tactics_df


def predict(clust_data, n_clusters=20):
    '''
    A function that returns predictions based on k-means clustering

    :param clust_data: [type: pandas dataframe]
    :param n_clusters: [type: int] - number of clusters for clustering
    :return: dataframe: [type: pandas dataframe] - tactical patterns of each cluster
    '''
    kmeans = KMeans(n_clusters, random_state=123)
    kmeans.fit(clust_data.values)
    y_pred = kmeans.predict(clust_data.values)
    return y_pred


def tactical_clust_patterns(clust_data, n_clusters=20):
    '''
    A function that returns a cluster dataframe showing different tactical similarities in each cluster
    The cluster dataframe is computed using k-means clustering (cluster_centers)

    :param clust_data: [type: pandas dataframe]
    :param n_clusters: [type: int] - number of clusters for clustering
    :return: dataframe: [type: pandas dataframe] - tactical patterns of each cluster
    '''
    kmeans = KMeans(n_clusters, random_state=123)
    kmeans.fit(clust_data.values)
    clust_kmeans = pd.DataFrame(kmeans.cluster_centers_, columns=clust_data.columns)
    clust_stats = clust_data.describe(percentiles=[0.1, 0.35, 0.65, 0.9])
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


def k_elbow_plot(csv_file):
    '''
    A function that creates a k-elbow plot to help identify optimal k value
    :param csv_file: [type: csv]
    :return:
    '''
    # k means determine k
    clust_data, _, _ = cluster_data_prep(csv_file)
    distortions = []
    cluster_range = range(1,30)
    for k in cluster_range:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(clust_data)
        distortions.append(sum(np.min(cdist(clust_data,
                        kmeanModel.cluster_centers_, 'euclidean'), axis=1))/clust_data.shape[0])
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
    kmeans_tactics_df = trend(csvFILEpath)
    k_elbow_plot(csvFILEpath)

