import pandas as pd
import pdfplumber
from pathlib import Path

def load_table(file_path: str) -> pd.DataFrame:
    ext = Path(file_path).suffix.lower().replace('.', '')
    if ext == "csv":
        return pd.read_csv(file_path)
    elif ext in ["xlsx", "xls"]:
        return pd.read_excel(file_path)
    elif ext == "json":
        return pd.read_json(file_path)
    elif ext == "pdf":
        text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text())
        return pd.DataFrame({"text": text})
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def infer_schema(df: pd.DataFrame) -> dict:
    return {col: {"dtype": str(df[col].dtype), "nulls": int(df[col].isnull().sum())} for col in df.columns}
