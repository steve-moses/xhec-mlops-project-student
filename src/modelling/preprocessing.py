import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(datapath: str) -> pd.DataFrame:
    """
    Load data from a given CSV path.

    Parameters:
    - datapath (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(datapath)

def preprocessing(df: pd.DataFrame) -> tuple:
    """
    Preprocess the given data. Extract age from Rings and split the data into train and test sets.

    Parameters:
    - df (pd.DataFrame): Input data as a DataFrame.

    Returns:
    - tuple: Contains X_train, X_test, y_train, y_test.
    """
    df['age'] = df['Rings'] + 1.5
    X = df.drop("Rings", axis=1).copy()
    y = X.pop("age")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
