import logging
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["title", "categories"]

def preprocess_dataframe(df: pd.DataFrame) -> List[str]:
    
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    text_column = None
    if "abstract" in df.columns:
        text_column = "abstract"
    elif "summary" in df.columns:
        text_column = "summary"
    else:
        raise ValueError("Dataset must contain either 'abstract' or 'summary'")

    logger.info(f"Using text column: {text_column}")

    df = df.fillna("")
    documents = (
        df["title"].str.lower() + ". " +
        df[text_column].str.lower() + ". " +
        "categories: " + df["categories"].str.lower()
    )

    return documents.tolist()
