

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import lightgbm
from sklearn import model_selection
import hyperopt
import logging
from hyperopt.pyll.base import scope


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def fit_and_log_cv(x_train,
                   y_train,
                   x_test,
                   y_test,
                   params,
                   nested):
  """Fit a model and log it along with train/CV metrics.

  Args:
      x_train: feature matrix for training/CV data
      y_train: label array for training/CV data
      x_test: feature matrix for test data
      y_test: label array for test data
      nested: if true, mlflow run will be started as child
          of existing parent
  """
  with mlflow.start_run(nested=nested) as run:
    # Fit CV models; extract predictions and metrics
    print(type(params))
    print(params)
    model_cv = lightgbm.LGBMClassifier(**params)
    y_pred_cv = model_selection.cross_val_predict(model_cv, x_train, y_train, method = "predict_proba")[:,1]
    metrics_cv = {
      f"val_{metric}": value
      for metric, value in eval_metrics(y_train, y_pred_cv).items()}

    # Fit and log full training sample model; extract predictions and metrics
    mlflow.lightgbm.autolog()
    dataset = lightgbm.Dataset(x_train, label=y_train)
    model = lightgbm.train(params=params, train_set=dataset)
    y_pred_test = model.predict(x_test)
    metrics_test = {
      f"test_{metric}": value
      for metric, value in eval_metrics(y_test, y_pred_test).items()}

    metrics = {**metrics_test, **metrics_cv}
    mlflow.log_metrics(metrics)
    return metrics


def eval_metrics(actual, predicted_prob):
    #f1 = f1_score(actual, predicted_prob)
    auc = 1 - roc_auc_score(actual, predicted_prob)
    return {"auc": auc}

def build_train_objective(x_train,
                          y_train,
                          x_test,
                          y_test,
                          metric):
    """Build optimization objective function fits and evaluates model.

    Args:
      x_train: feature matrix for training/CV data
      y_train: label array for training/CV data
      x_test: feature matrix for test data
      y_test: label array for test data
      metric: name of metric to be optimized

    Returns:
        Optimization function set up to take parameter dict from Hyperopt.
    """

    def train_func(params):
        """Train a model and return loss metric."""
        metrics = fit_and_log_cv(
          x_train, y_train, x_test, y_test, params, nested=True)
        return {'status': hyperopt.STATUS_OK, 'loss': metrics[metric]}

    return train_func



def log_best(run: mlflow.entities.Run,
        metric: str) -> None:
    """Log the best parameters from optimization to the parent experiment.

    Args:
        run: current run to log metrics
        metric: name of metric to select best and log
    """

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        [run.info.experiment_id],
        "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id))

    best_run = min(runs, key=lambda run: run.data.metrics[metric])

    mlflow.set_tag("best_run", best_run.info.run_id)
    mlflow.log_metric(f"best_{metric}", best_run.data.metrics[metric])


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    mlflow.end_run()
    df = pd.read_csv("bn_training_data.csv")
    dropped_cols = [
        "box_code",
        "bank_code",
        "plausi_id",
        "first_round",
        "annotation",
        "is_reversal",
        "period",
        "calc_forecast_auto_arima"
        # "calc_forecast_holt",
        # "calc_forecast_ses",
        # "calc_forecast_mean"
    ]

    Xtrain, Xtest, ytrain, ytest = train_test_split(df.drop(dropped_cols + ["second_round"], axis = 1), 
                                                    df["second_round"], 
                                                    test_size = 0.25, 
                                                    random_state = 1)

    
    space = {
        'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.5, 1.0),
        'subsample': hyperopt.hp.uniform('subsample', 0.05, 1.0),
        # The parameters below are cast to int using the scope.int() wrapper
        'num_iterations': scope.int(
        hyperopt.hp.quniform('num_iterations', 10, 200, 1)),
        'num_leaves': scope.int(hyperopt.hp.quniform('num_leaves', 20, 50, 1))
    }

    with mlflow.start_run() as run:
        hyperopt.fmin(fn=build_train_objective(Xtrain, ytrain, Xtest, ytest, "val_auc"),
                        space=space,
                        algo=hyperopt.tpe.suggest,
                        max_evals=10)
        log_best(run, "val_auc")
        search_run_id = run.info.run_id
        experiment_id = run.info.experiment_id
