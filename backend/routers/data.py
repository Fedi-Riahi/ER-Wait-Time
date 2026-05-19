from fastapi import APIRouter, UploadFile, File, HTTPException
import sys, os, io, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data_loader   import load_raw
from preprocessing import clean
from config        import DATASET_PATH

router = APIRouter()


@router.get("/preview")
def preview():
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found at '{DATASET_PATH}'. Check config.DATASET_PATH.",
        )
    df   = load_raw()
    df   = clean(df)
    cols = list(df.columns)
    return {
        "rows":       len(df),
        "columns":    len(cols),
        "col_names":  cols,
        "null_count": int(df.isnull().sum().sum()),
        "preview":    df.head(10).fillna("NULL").to_dict(orient="records"),
        "dataset":    DATASET_PATH,   # so the frontend can show which file is loaded
    }


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode()))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    return {
        "rows":       len(df),
        "columns":    len(df.columns),
        "col_names":  list(df.columns),
        "null_count": int(df.isnull().sum().sum()),
        "preview":    df.head(10).fillna("NULL").to_dict(orient="records"),
    }


@router.get("/schema")
def schema():
    """Return column names + dtypes — useful for building dynamic predict forms."""
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    df = load_raw()
    df = clean(df)
    return {
        "columns": [
            {"name": col, "dtype": str(df[col].dtype)}
            for col in df.columns
        ]
    }
