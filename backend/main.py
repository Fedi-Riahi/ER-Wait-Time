
import sys
import os

# Allow backend to import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import train, experiments, data, predict, mlflow_runs

app = FastAPI(title="ER Wait Time — ML Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train.router,        prefix="/train",       tags=["Training"])
app.include_router(experiments.router,  prefix="/experiments", tags=["Experiments"])
app.include_router(data.router,         prefix="/data",        tags=["Data"])
app.include_router(predict.router,      prefix="/predict",     tags=["Predict"])
app.include_router(mlflow_runs.router,  prefix="/mlflow",      tags=["MLflow"])

@app.get("/health")
def health():
    return {"status": "ok"}
