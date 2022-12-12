# call functions and config modules
# exec(open("./functions/functions.py").read())
# exec(open("./config/config.py").read())
import numpy as np
# Basics
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score
# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
# Pipeline
# Scaler for standardization
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, features, method='constant', value='missing'):
        self.features = features
        self.method = method
        self.value = value

    def fit(self, X, y=None):
        if self.method == 'mean':
            self.value = X[self.features].mean()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = X[self.features].fillna(self.value)
        return X_transformed


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.min = X[self.features].min()
        self.range = X[self.features].max() - self.min
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = (X[self.features] - self.min) / self.range
        return X_transformed


class NumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.powertransform = PowerTransformer(standardize=True)
        self.powertransform.fit(X[self.features])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        power = PowerTransformer(standardize=True)
        X_transformed[self.features] = power.fit_transform(X_transformed[self.features])
        return X_transformed


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop='first'):
        self.features = features
        self.drop = drop

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse=False, drop=self.drop, handle_unknown="ignore")
        self.encoder.fit(X[self.features])
        return self

    def transform(self, X):
        X_transformed = pd.concat([X.drop(columns=self.features).reset_index(drop=True),
                                   pd.DataFrame(self.encoder.transform(X[self.features]),
                                                columns=self.encoder.get_feature_names(self.features))],
                                  axis=1)
        return X_transformed


class MultiColumnLabelEncoder:
    def __init__(self, features, drop='first'):
        self.features = features
        self.drop = drop

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        X_transformed = X.copy()
        if self.features is not None:
            for col in self.features:
                X_transformed[col] = LabelEncoder().fit_transform(X_transformed[col])
        else:
            for colname, col in X_transformed.iteritems():
                X_transformed[colname] = LabelEncoder().fit_transform(col)
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def calculate_roc_auc(model_pipe, X, y):
    """Calculate roc auc score.

    Parameters:
    ===========
    model_pipe: sklearn model or pipeline
    X: features
    y: true target
    """
    y_proba = model_pipe.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_proba)
