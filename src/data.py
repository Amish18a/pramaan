from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "isot"

def load_isot():
    fake = pd.read_csv(RAW / "Fake.csv")
    fake["label"] = 1 
    real = pd.read_csv(RAW / "True.csv")
    real["label"] = 0
    df = pd.concat([fake,real],ignore_index=True)

    df["date_raw"] = df["date"]
    df["date"] = pd.to_datetime(df["date"],errors="coerce")
    return df