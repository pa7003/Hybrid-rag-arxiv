def preprocess_dataframe(df):
    df = df.fillna("")
    documents = (
        df["title"].str.lower() + ". " +
        df["abstract"].str.lower() + ". " +
        "categories: " + df["categories"].str.lower()
    )
    return documents.tolist()
