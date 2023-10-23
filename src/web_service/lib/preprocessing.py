import numpy as np
import pandas as pd


def get_num_cat_cols(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns
    return num_cols, cat_cols
