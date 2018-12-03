import math
import numpy as np
import pandas as pd
from itertools import cycle

from preprocessor import feat_eng, create_model_df, category_encoder

from scipy import interp
import scipy.stats as scs
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, label_binarize


from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as PreRecF1Support_score
from sklearn.metrics import classification_report


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt


class SoccerModel(object):
    '''
    Model for soccer predictions for supervised/unsupervised learning.
    '''
    def __init__(self, data_path):
        self.data_path = data_path
        self.columns = None

    def get_data(self):
        '''
        Preprocess csv file through dataframe.
        Create features set X.
        Create targets set y.
        '''
        df = create_model_df(self.data_path)
        y = df.pop('resultsLabel')
        X = df.values
        return X, y

    def fit(self, X_train, y_train, model):
        '''
        Fit model classifier with training data.
        '''
        self.model = model
        self.model.fit(X_train, y_train)

    def predict_proba(self, X_test):
        '''
        Returns predicted probabilities for targets [-1, 0, 1] --> [loss, draw, win].
        '''
        return self.model.predict_proba(X_test)[:, 1]

    def predict(self, X_test):
        '''
        Returns predictions for the defined model
        '''
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        '''
        Returns accuracy score for predictions
        '''
        return self.model.score(X_test, y_test)


if __name__ == '__main__':
    csvFILEpath = input("Enter path to file that you wish to pre-process: (should be a .csv file) --> "
                        "../data/FootballEurope/FootballEurope.csv ")
    model = SoccerModel(csvFILEpath)
    X, y = model.get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                    test_size=0.2, random_state=123, stratify=y)
    rfc = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
    model.fit(X_train, y_train, rfc)
    y_pred = model.predict(X_test)
    print()
    print('Soccer model accuracy: {0:0.2f}%'.format(100*model.score(X_test, y_test)))
    C = confusion_matrix(y_test, y_pred)
    Cnorm = np.around((100 * C / C.astype(np.float).sum(axis=1)), decimals=1)
    # print('Confusion Matrix:\n {}'.format(C))
    print()
    print('Confusion Matrix Normalized:\n {}'.format(Cnorm))
    print()
    print('      Confusion Matrix:')
    print(pd.crosstab(y_test, y_pred, rownames=['True'],
            colnames=['Predicted'], margins=True))

