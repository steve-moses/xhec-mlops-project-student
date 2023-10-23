# Pydantic models for the web service
from pydantic import BaseModel
from typing import List


class InputData(BaseModel):
    Sex: str
    Length: float
    Diameter: float
    Height: float
    Whole: float
    Shucked: float
    Viscera: float
    Shell: float
