# Code with FastAPI (app = FastAPI(...))
from app_config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    PATH_TO_MODEL,
)
from fastapi import FastAPI
from utils import load_model
from lib.inference import run_inference
from lib.models import InputData

# Other imports

app = FastAPI(title="...", description="...")


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", status_code=201)
def predict(payload: InputData):
    model = load_model(PATH_TO_MODEL)
    y = run_inference([payload], model)
    print("THIS IS THE ABALONE AGE:", y)
    return {"The abalone age is": y[0]}
