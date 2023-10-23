"""
This module provides functionality to save Python objects to disk using Python's pickle module. 
It's particularly useful for serializing and saving machine learning models for later use.
"""

from typing import Any
import os
import pickle


def save_pickle(path: str, obj: Any) -> None:
    """
    Serialize and save a Python object to disk using pickle.

    Parameters:
    - path (str): The path where the pickled object should be saved.
    - obj (Any): The Python object to be serialized and saved.

    Returns:
    - None
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
