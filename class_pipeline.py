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
from sklearn.pipeline import Pipeline
import pycaret
from pycaret.classification import *

pd.set_option('display.max_columns', 500)


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, features, method='constant', value='missing'):
        self.features = features
        self.method = method
        self.value = value

    def fit(self, X_train, y=None):
        if self.method == 'mean':
            self.value = X[self.features].mean()
        return self

    def transform(self, X_train):
        X_transformed = X_train.copy()
        X_transformed[self.features] = X[self.features].fillna(self.value)
        return X_transformed


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X_train, y=None):
        self.min = X_train[self.features].min()
        self.range = X_train[self.features].max() - self.min
        return self

    def transform(self, X_train):
        X_transformed = X_train.copy()
        X_transformed[self.features] = (X_train[self.features] - self.min) / self.range
        return X_transformed


class NumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X_train, y=None):
        self.powertransform = PowerTransformer(standardize=True)
        self.powertransform.fit(X[self.features])
        return self

    def transform(self, X_train):
        X_transformed = X_train.copy()
        power = PowerTransformer(standardize=True)
        X_transformed[self.features] = power.fit_transform(X_transformed[self.features])
        return X_transformed


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop='first'):
        self.features = features
        self.drop = drop

    def fit(self, X_train, y=None):
        self.encoder = OneHotEncoder(sparse=False, drop=self.drop, handle_unknown="ignore")
        self.encoder.fit(X[self.features])
        return self

    def transform(self, X_train):
        X_transformed = pd.concat([X_train.drop(columns=self.features).reset_index(drop=True),
                                   pd.DataFrame(self.encoder.transform(X_train[self.features]),
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


# %%
# read in the credit risk data set
credit_risk = pd.read_csv("./data/credit_risk_data.csv")

# reset index to ID
credit_risk = credit_risk.set_index('SK_ID_CURR')

colnames = list(credit_risk.columns.drop('TARGET'))  # create list of predictor names
categorical_names = credit_risk.select_dtypes(include=[object, np.int64]).columns  # create list of categorical names
continuous_names = credit_risk.select_dtypes(exclude=[object, np.int64]).columns  # create list of continuous names

# %%
# define features (X) and target (y)
X = credit_risk[colnames]  # X value contains all the variables except labels
y = credit_risk['TARGET']  # these are the labe'

SEED = 42
TARGET = 'TARGET'
FEATURES = credit_risk.columns.drop(TARGET)

NUMERICAL = credit_risk[FEATURES].select_dtypes('number').columns
CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))

credit_risk = credit_risk.dropna()
X_train, X_test, y_train, y_test = train_test_split(credit_risk.drop(columns=TARGET), credit_risk[TARGET],
                                                    test_size=.2, random_state=SEED,
                                                    stratify=credit_risk[TARGET])
# fit data only

data_pipeline = Pipeline(steps=[['num_imputer', Imputer(NUMERICAL, method='mean')],
                                ['scaler', NumericTransformer(NUMERICAL)],
                                ['cat_imputer', Imputer(CATEGORICAL)],
                                ['encoder', MultiColumnLabelEncoder(CATEGORICAL)]])

X_train_processed = data_pipeline.transform(X_train)
X_test_processed = data_pipeline.transform(X_test)

# XGB -----------------------------------------------------------------------------------------------------------------
pipeline = imbpipeline(steps=[['num_imputer', Imputer(NUMERICAL, method='mean')],
                              ['scaler', NumericTransformer(NUMERICAL)],
                              ['cat_imputer', Imputer(CATEGORICAL)],
                              ['encoder', MultiColumnLabelEncoder(CATEGORICAL)],
                              ['smote', SMOTE(random_state=11)],
                              ['model', xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)]])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
sum(preds)
sum(y_train)

print("The accuracy of XGBoost model is:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
print("The precision score is: ", precision_score(y_true=y_test, y_pred=preds, average="binary"))
print("The recall score is: ", recall_score(y_true=y_test, y_pred=preds, average="binary"))

print(f"Train ROC-AUC: {calculate_roc_auc(pipeline, X_train, y_train):.4f}")
print(f"Test ROC-AUC: {calculate_roc_auc(pipeline, X_test, y_test):.4f}")


# LR -----------------------------------------------------------------------------------------------------------------
pipeline = imbpipeline(steps=[['num_imputer', Imputer(NUMERICAL, method='mean')],
                              ['scaler', NumericTransformer(NUMERICAL)],
                              ['cat_imputer', Imputer(CATEGORICAL)],
                              ['encoder', MultiColumnLabelEncoder(CATEGORICAL)],
                              ['smote', SMOTE(random_state=11)],
                              ['model', LogisticRegression()]])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print("The accuracy of XGBoost model is:", accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
print("The precision score is: ", precision_score(y_true=y_test, y_pred=preds, average="binary"))
print("The recall score is: ", recall_score(y_true=y_test, y_pred=preds, average="binary"))

print(f"Train ROC-AUC: {calculate_roc_auc(pipeline, X_train, y_train):.4f}")
print(f"Test ROC-AUC: {calculate_roc_auc(pipeline, X_test, y_test):.4f}")


# KNN -----------------------------------------------------------------------------------------------------------------
pipeline = imbpipeline(steps=[['num_imputer', Imputer(NUMERICAL, method='mean')],
                              ['scaler', NumericTransformer(NUMERICAL)],
                              ['cat_imputer', Imputer(CATEGORICAL)],
                              ['encoder', MultiColumnLabelEncoder(CATEGORICAL)],
                              ['smote', SMOTE(random_state=11)],
                              ['model', KNeighborsClassifier(n_neighbors=5)]])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print("The accuracy of XGBoost model is:", accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
print("The precision score is: ", precision_score(y_true=y_test, y_pred=preds, average="binary"))
print("The recall score is: ", recall_score(y_true=y_test, y_pred=preds, average="binary"))

print(f"Train ROC-AUC: {calculate_roc_auc(pipeline, X_train, y_train):.4f}")
print(f"Test ROC-AUC: {calculate_roc_auc(pipeline, X_test, y_test):.4f}")


# rf -----------------------------------------------------------------------------------------------------------------
pipeline = imbpipeline(steps=[['num_imputer', Imputer(NUMERICAL, method='mean')],
                              ['scaler', NumericTransformer(NUMERICAL)],
                              ['cat_imputer', Imputer(CATEGORICAL)],
                              ['encoder', MultiColumnLabelEncoder(CATEGORICAL)],
                              ['smote', SMOTE(random_state=11)],
                              ['model', RandomForestClassifier(max_depth=10, max_features=6, n_estimators=100)]])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print("The accuracy of XGBoost model is:", accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
print("The precision score is: ", precision_score(y_true=y_test, y_pred=preds, average="binary"))
print("The recall score is: ", recall_score(y_true=y_test, y_pred=preds, average="binary"))

print(f"Train ROC-AUC: {calculate_roc_auc(pipeline, X_train, y_train):.4f}")
print(f"Test ROC-AUC: {calculate_roc_auc(pipeline, X_test, y_test):.4f}")


# ---------------------------------------------------------------------------------------------------------------------

# Create the parameter grid based on the results of random search
pipeline = imbpipeline(steps=[['num_imputer', Imputer(NUMERICAL, method='mean')],
                              ['scaler', NumericTransformer(NUMERICAL)],
                              ['cat_imputer', Imputer(CATEGORICAL)],
                              ['encoder', MultiColumnLabelEncoder(CATEGORICAL)],
                              ['smote', SMOTE(random_state=11)],
                              ['model', xgb.XGBClassifier(objective='binary:logistic', seed=123)]])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
# Create the parameter grid
gbm_param_grid = {
    'model__learning_rate': np.arange(0.05, 1, 0.05),
    'model__max_depth': np.arange(3, 10, 1),
    'model__n_estimators': np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator=pipeline, n_iter=1, scoring="accuracy", verbose=1,
                                        param_distributions=gbm_param_grid)
# Fit the estimator
randomized_roc_auc.fit(X_train, y_train)

# Compute metrics
print("Best ROC: ", np.sqrt(np.abs(randomized_roc_auc.best_score_)))
print("Best model: ", randomized_roc_auc.best_estimator_)

# randomized_roc_auc.best_params_

# Create the parameter grid based on the results of random search
pipeline = imbpipeline(steps=[['num_imputer', Imputer(NUMERICAL, method='mean')],
                              ['scaler', NumericTransformer(NUMERICAL)],
                              ['cat_imputer', Imputer(CATEGORICAL)],
                              ['encoder', MultiColumnLabelEncoder(CATEGORICAL)],
                              ['smote', SMOTE(random_state=11)],
                              ['model', xgb.XGBClassifier(objective='binary:logistic',
                                                          seed=123,
                                                          n_estimator=150,
                                                          max_depth=5,
                                                          learning_rate=0.6)]])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print("The accuracy of XGBoost model is:", accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
print("The precision score is: ", precision_score(y_true=y_test, y_pred=preds, average="binary"))
print("The recall score is: ", recall_score(y_true=y_test, y_pred=preds, average="binary"))

print(f"Train ROC-AUC: {calculate_roc_auc(pipeline, X_train, y_train):.4f}")
print(f"Test ROC-AUC: {calculate_roc_auc(pipeline, X_test, y_test):.4f}")


# ---------------------------------------------------------------------------------------------------------------------
# pycaret

preprocessed_train = X_train_processed.merge(y_train, how="left", left_index=True, right_index=True)
preprocessed_test = X_test_processed.merge(y_test, how="left", left_index=True, right_index=True)

s = setup(
    fold=10,  # 10 fold cross validation
    data=preprocessed_train,  # training data
    silent=True,
    test_data=preprocessed_test,  # test data
    target='TARGET',
    session_id=123)

# %%
# display basic logistic regression model on raw data
best = compare_models(include=['lr'])
lr_fit = create_model('lr', fold=5)
plot_model(lr_fit, plot='confusion_matrix', plot_kwargs={'percent': False})
plot_model(lr_fit, 'boundary')

# %%
clf1 = setup(fold=2,  # 10 fold cross validation
             data=preprocessed_train,  # training data
             silent=True,
             test_data=preprocessed_test,  # test data
             target='TARGET',
             fix_imbalance=True,
             remove_outliers=True,
             outliers_threshold=0.025,
             normalize=False,
             log_experiment=True,
             remove_multicollinearity=True,
             multicollinearity_threshold=0.6)  # transformation = True

best = compare_models(include=['lr', 'knn', 'rf', 'dt', 'xgboost'],
                      sort='F1')
