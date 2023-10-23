"""
This module contains the main training flow for a machine learning model.
It reads the data, preprocesses it, trains a model, and then saves it.
"""

import argparse
import os
from typing import Optional

import mlflow
from predicting import predict
from preprocessing import load_data, preprocessing
from training import evaluate_model, train_model
from utils import save_pickle


def main(trainset_path: str, artifacts_filepath: Optional[str] = None) -> float:
    """
    Train a model using data from the provided path and save it in pickle format.

    Parameters:
        trainset_path (str): Path to the training data set.
        artifacts_filepath (Optional[str]): Path to save the trained model. Defaults to None.

    Returns:
        float: Root Mean Squared Error of the trained model.
    """
    mlflow.set_experiment("Abalone_Age_Prediction")
    with mlflow.start_run() as run:
        mlflow.sklearn.autolog()
        run_id = run.info.run_id

        # Set tags for the run
        mlflow.set_tag("release.version", "0.0.0")

        # Read and preprocess data
        data = load_data(trainset_path)
        X_train, X_test, y_train, y_test = preprocessing(data)

        # Train model and make predictions
        model = train_model(X_train, y_train)
        mlflow.log_params(model.get_params())
        y_predict = predict(X_test, model)

        # Save the model in pickle format
        if not artifacts_filepath:
            artifacts_filepath = "src/web_service/local_objects"
        save_pickle(os.path.join(artifacts_filepath, "model.pkl"), model)

        # Evaluate the model and log the metrics
        rmse = evaluate_model(y_test, y_predict)
        mlflow.log_metric("Root Mean Squared Error", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.register_model(f"runs:/{run_id}/models", "model")
        current_tracking_uri = mlflow.get_tracking_uri()
        print(current_tracking_uri)

    return print(rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    # parser.add_argument("artifacts_filepath", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
