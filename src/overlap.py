import re
import pandas as pd

def fingerprint(s, a=100, b=250):
    n = re.sub(r"[^a-z0-9]", "", str(s).lower())
    return n[a:b] if len(n) >= b else None

def drop_overlap(candidate, reference, min_words=10):
    ref_keys = {k for k in reference["text"].map(fingerprint) if isinstance(k, str)}
    c = candidate.copy()
    c["key"] = c["text"].map(fingerprint)
    c = c[c["key"].notna() & ~c["key"].isin(ref_keys)]   # checked AND not in reference
    c = c[c["text"].str.split().str.len() >= min_words]
    c = c.drop_duplicates(subset=["text"])
    return c.drop(columns="key").reset_index(drop=True)