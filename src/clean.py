import re

def clean_isot(df, min_words=10):
    df = df.copy()
    # 1. Drop near-empty texts (866 of these are fake, a collection artifact)
    df["words"] = df["text"].str.split().str.len()
    df = df[df["words"] >= min_words]
    # 2. Drop exact duplicates BEFORE splitting into train/test
    df = df.drop_duplicates(subset=["title", "text"])
    # 5. Drop malformed scraped rows whose date could not be parsed
    df = df[df["date"].notna()]
    # 3. Remove the Reuters dateline at the start, e.g. "WASHINGTON (Reuters) - "
    df["text"] = df["text"].str.replace(r"^[^()]{0,60}\(Reuters\)\s*-?\s*", "", regex=True)
    df["text"] = df["text"].str.replace("(Reuters)", " ", regex=False)
    # 4. Remove all quote/apostrophe characters (fake text has none, real has curly ones)
    df["text"] = df["text"].str.replace(r"[\"'‘’“”`]", " ", regex=True)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df.reset_index(drop=True)

EXTRA_PATTERNS = {
    "reuters_word": r"Reuters",
    "ticker": r"\([A-Z]{1,5}\.[A-Z]{1,2}\)",
    "signoff": r"(Reporting|Editing|Writing) by[^.]*",
    "url": r"https?://\S+|www\.\S+",
    "handle": r"@\w+",
}

def clean_extra(df, groups=None):
    """Remove extra shortcut patterns. groups=None removes all of them."""
    df = df.copy()
    for name, pat in EXTRA_PATTERNS.items():
        if groups is None or name in groups:
            df["text"] = df["text"].str.replace(pat, " ", regex=True)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df