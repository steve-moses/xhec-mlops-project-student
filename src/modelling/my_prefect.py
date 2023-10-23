import os

import mlflow
import mlflow.sklearn
from prefect import flow, task
from preprocessing import load_data, preprocessing
from training import evaluate_model, train_model
from utils import save_pickle

mlflow.set_experiment("Ablone_prediction")
mlflow.sklearn.autolog()


@task
def load_data_task(trainset_path: str):
    return load_data(trainset_path)


@task
def preprocessing_task(data):
    return preprocessing(data)


@task
def train_model_task(X_train, y_train):
    return train_model(X_train, y_train)


@task
def log_mlflow(model, X_test, y_test):
    with mlflow.start_run():
        y_pred = model.predict(X_test)
        rmse = evaluate_model(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model")


@task
def save_model(export_path: str, obj):
    save_pickle(export_path, obj)


@flow(name="albone_prediction")
def main_flow():
    data = load_data_task(os.path.join(os.getcwd(), "..", "..", "data", "ablone.csv"))
    X_train, X_test, y_train, y_test = preprocessing_task(data)
    model = train_model_task(X_train, y_train)
    log_mlflow(model, X_test, y_test)
    save_model(os.path.join(os.getcwd(), "model.pkl"), model)


if __name__ == "__main__":
    main_flow.serve(name="final-deployment")
