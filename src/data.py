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
    df["date"] = pd.to_datetime(df["date_raw"].str.strip(), format="mixed", errors="coerce")
    return df

WELFAKE = ROOT / "data" / "raw" / "welfake" / "WELFake_Dataset.csv"

def load_welfake():
    df = pd.read_csv(WELFAKE)
    # drop the unnamed serial-number column if present
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df = df.rename(columns={"label": "label_raw"})   # NOT mapped yet
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    return df