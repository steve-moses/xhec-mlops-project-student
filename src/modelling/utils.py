# Use this module to code a `pickle_object` function. This will be useful to pickle the model (and encoder if need be).
from typing import Any

import os
import pickle

def load_pickle(path: str):
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pickle(path: str, obj: Any):
    with open(path, "wb") as f:
        pickle.dump(obj, f)