# MLFlow python demo code

This code tunes some hyperparameters using hyperopt and logs all results to mlflow.
Running `fit_model.py` will do the hyperparameter search and log all results in the folder `mlruns/0`.
You can then view everything in the web ui by running `mlflow ui` in the command line from within the folder.

You can serve any of the models as local rest api by running the command in `useful_commands.txt` in the command line and pointing it at the right id. Note you will need to change the `conda.yaml` to use artifactory (as in your conda.rc). 