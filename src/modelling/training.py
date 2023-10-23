from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.metrics import mean_squared_error


def train_model(X, y):
    """
    Train a linear regression model on the provided data.

    The function identifies numerical and categorical columns, preprocesses them using a 
    pipeline (scaling numerical columns and one-hot encoding categorical ones), and then trains a 
    linear regression model.

    Parameters:
    - X: Features/dataframe.
    - y: Target/labels.

    Returns:
    - pipeline: A trained pipeline that includes preprocessing and the model.
    """
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include="object").columns
    pipeline = get_pipeline(num_cols, cat_cols)
    pipeline.fit(X, y)
    return pipeline


def get_pipeline(numerical_cols, categorical_cols):
    """
    Create a pipeline for preprocessing and linear regression.

    The pipeline scales numerical columns and one-hot encodes categorical columns, then applies 
    linear regression.

    Parameters:
    - numerical_cols: List of numerical column names.
    - categorical_cols: List of categorical column names.

    Returns:
    - pipeline: A pipeline with preprocessing and linear regression steps.
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    model = LinearRegression()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    return pipeline


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Evaluate the model's performance using the Root Mean Squared Error (RMSE).

    Parameters:
    - y_true (np.ndarray): True target values.
    - y_pred (np.ndarray): Predicted target values by the model.

    Returns:
    - float: RMSE of the model's predictions.
    """
    return mean_squared_error(y_true, y_pred, squared=False)
