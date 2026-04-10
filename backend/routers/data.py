
from fastapi import APIRouter, UploadFile, File
import sys, os, io, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from data_loader   import load_raw, load_from_bytes
from preprocessing import clean

router = APIRouter()

@router.get("/preview")
def preview():
    df   = load_raw()
    df   = clean(df)
    cols = list(df.columns)
    return {
        "rows":      len(df),
        "columns":   len(cols),
        "col_names": cols,
        "null_count": int(df.isnull().sum().sum()),
        "preview":   df.head(10).fillna("NULL").to_dict(orient="records"),
    }

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    df      = pd.read_csv(io.StringIO(content.decode()))
    return {
        "rows":      len(df),
        "columns":   len(df.columns),
        "col_names": list(df.columns),
        "null_count": int(df.isnull().sum().sum()),
        "preview":   df.head(10).fillna("NULL").to_dict(orient="records"),
    }
