import pandas as pd
from src.preprocessing import preprocess_dataframe

def test_preprocess_dataframe():
    data = {
        "title": ["Test Paper"],
        "abstract": ["This is a TEST abstract."],
        "categories": ["cs.AI"]
    }
    df = pd.DataFrame(data)

    documents = preprocess_dataframe(df)

    assert len(documents) == 1
    assert "test paper" in documents[0]
    assert "test abstract" in documents[0]
