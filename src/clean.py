import re

def clean_isot(df, min_words=10):
    df = df.copy()
    # 1. Drop near-empty texts (866 of these are fake, a collection artifact)
    df["words"] = df["text"].str.split().str.len()
    df = df[df["words"] >= min_words]
    # 2. Drop exact duplicates BEFORE splitting into train/test
    df = df.drop_duplicates(subset=["title", "text"])
    # 3. Remove the Reuters dateline at the start, e.g. "WASHINGTON (Reuters) - "
    df["text"] = df["text"].str.replace(r"^[^()]{0,60}\(Reuters\)\s*-?\s*", "", regex=True)
    df["text"] = df["text"].str.replace("(Reuters)", " ", regex=False)
    # 4. Remove all quote/apostrophe characters (fake text has none, real has curly ones)
    df["text"] = df["text"].str.replace(r"[\"'‘’“”`]", " ", regex=True)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df.reset_index(drop=True)