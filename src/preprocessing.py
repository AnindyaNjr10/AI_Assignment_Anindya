"""Data loading, cleaning and text preprocessing (Tasks 1)."""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tickets.csv"


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/special characters, collapse extra spaces.

    Stop-words are deliberately NOT removed: words like "not" and "cannot"
    carry meaning in support tickets ("password reset is NOT working").
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation / symbols
    text = re.sub(r"\s+", " ", text).strip()   # remove extra whitespace
    return text


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Handle missing values and duplicates, then add a `clean_description` column."""
    df = df.copy()
    n0 = len(df)

    # 1. Missing values --------------------------------------------------
    # A ticket without a description (or category) cannot be used for training -> drop.
    df = df.dropna(subset=["ticket_description", "category"])
    # priority / status are not model inputs -> keep the row, label as Unknown.
    df["priority"] = df["priority"].fillna("Unknown")
    df["status"] = df["status"].fillna("Unknown")

    # 2. Text cleaning ---------------------------------------------------
    df["clean_description"] = df["ticket_description"].apply(clean_text)
    df["category"] = df["category"].str.strip()

    # 3. Duplicates (same cleaned text + same category) ------------------
    before = len(df)
    df = df.drop_duplicates(subset=["clean_description", "category"]).reset_index(drop=True)

    if verbose:
        print(f"Rows before cleaning            : {n0}")
        print(f"Dropped (missing description)   : {n0 - before}")
        print(f"Dropped (duplicates)            : {before - len(df)}")
        print(f"Rows after cleaning             : {len(df)}")
    return df
