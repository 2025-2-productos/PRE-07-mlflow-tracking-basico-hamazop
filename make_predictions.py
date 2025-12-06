
"""Prediction script for the MLflow model.

This script loads a model from MLflow and makes predictions on a dataset

python make_predictions.py
"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

## Debe verificarse el run_id del modelo que se quiere cargar
## Se puede obtenerel run_id desdela interfaz de MLflow

logged_model = "runs:/fb352fd4e37442bbb436edd7dd963c78/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)

print(y)