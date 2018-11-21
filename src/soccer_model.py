import math
import numpy as np
import pandas as pd
from itertools import cycle

from preprocesser import feat_eng, category_encoder

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
%matplotlib inline
%config InlineBackend.figure_format='retina'


class Model(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.columns = None

    def get_data(self):
        '''
        Create dataframe from raw json file
        Create features set X
        Create targets set y
        '''
        json_df = read_json(self.data_path)
        df = preprocess_feat_eng(json_df)
        df_rf = df.drop(['description', 'name', 'org_desc', 'org_name'], axis=1)
        y = df_rf.pop('fraud')
        X = df_rf.values
        return X, y

    def fit(self, X_train, y_train):
        '''
        Fit Gradient Boosted Classifier with training data
        '''
        self.model = RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1)
        self.model.fit(X_train, y_train)

    def predict_proba(self, X_test):
        '''
        Returns predicted probabilities for not fraud / fraud
        '''
        return self.model.predict_proba(X_test)[:, 1]

    def predict(self, X_test):
        '''
        Returns predicted probabilities for not fraud / fraud
        '''
        return self.model.predict(X_test)


if __name__ == '__main__':
    csvFILEpath = Input("Enter path to file that you wish to preprocess: (should be a .csv file) ")
    model = Model(csvFILEpath)
    X, y = model.get_data()
    model.fit(X, y)