
from fastapi import APIRouter
from .mlflow_runs import get_runs

router = APIRouter()

@router.get("/")
def list_experiments():
    return get_runs()
