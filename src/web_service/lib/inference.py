from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from lib.models import InputData
from lib.preprocessing import get_num_cat_cols
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def run_inference(input_data: List[InputData], model: BaseEstimator) -> np.ndarray:
    """
    """
    logger.info(f"Running inference on:\n{input_data}")
    df = pd.DataFrame([x.dict() for x in input_data])
    df.rename(columns={"Whole":"Whole weight", 
                         "Shucked":"Shucked weight", 
                         "Viscera":"Viscera weight", 
                         "Shell":"Shell weight"}, inplace=True)
    numerical_cols, categorical_cols = get_num_cat_cols(df)
    y = model.predict(df)
    logger.info(f"Predicted trip durations:\n{y}")
    return y