import sys

import pandas as pd
import numpy as np
from pygments.lexer import default
from sklearn.tree import DecisionTreeClassifier
import pickle
import mlflow
import mlflow.sklearn
import dagshub
import logging


def model_build(x_train,y_train):
    if sys.argv:
        m = DecisionTreeClassifier(min_samples_split=2, max_depth=3)
        a=2,
        b=3
    else:
        m= DecisionTreeClassifier()
        a="default"
        b="default"

    m.fit(x_train, y_train)
    return m,a,b


if __name__ == "__main__":
    x_train = pd.read_csv("Data/feature/x_train.csv")
    y_train = pd.read_csv("Data/feature/y_train.csv")
    x_test = pd.read_csv("Data/feature/x_test.csv")
    y_test = pd.read_csv("Data/feature/y_test.csv")
    with mlflow.start_run(run_name="model_build"):
       model,a,b = model_build(x_train,y_train)
       score = model.score(x_test, y_test)
       mlflow.log_param("min_samples_split",a)
       mlflow.log_param("max_depth",b)
       mlflow.log_metric("accuracy",score)
       uri = ""
       mlflow.set_tracking_uri(uri)
       pickle.dump(model, open("model.pkl", "wb"))

