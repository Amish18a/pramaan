# Pramaan (प्रमाण)
Evidence-first fake news verification. Not just "real or fake": which claims,
what evidence, and how confident.

## Status
Milestone 1 complete: leakage audit, baseline classifiers and a
cross-dataset test. Next: Milestone 2 (evidence-based verification).

## Why
A classifier alone can't establish truth. It can learn writing style or
publisher artifacts. This project tests how well simple models generalize
and builds toward evidence-based verification.

## Key results (Milestone 1)
Full details, caveats and what we cannot conclude: see `FINDINGS.md`.

- In the ISOT dataset the label is almost the same thing as the publisher
  (Reuters vs. websites flagged as unreliable). A one-word rule ("no
  (Reuters) tag means fake") reaches macro-F1 0.996 on raw ISOT with no
  machine learning.
- After removing the shortcuts we could identify, a TF-IDF + Logistic
  Regression model still scores macro-F1 0.984 on held-out ISOT articles
  (mean over 5 seeds: raw 0.9915, cleaned 0.9849, cleaned + extra 0.9843).
- On 22,779 WELFake articles that do not overlap ISOT, the same model scores
  about 0.57 macro-F1 (Linear SVM: 0.57). Accuracy (58%) is about the same
  as always predicting "real" (58.4%), and 63% of real articles are labelled
  fake. ROC AUC is 0.72 to 0.75, so some signal transfers but the decision
  threshold does not.
- About 59% of WELFake overlapped ISOT, so only non-overlapping rows are used.
- The cause of the drop is not isolated: publisher style, topic, time and
  label quality all differ between the datasets.

| Check (ISOT) | Real | Fake |
|---|---|---|
| Text contains "(Reuters)" | 99.2% | 0.04% |
| Text contains curly quotes | 50.5% | 0.0% |
| Texts under 10 words | 1 | 866 |
| Exact duplicate rows | 220 | 5,573 |
| `subject` values | politicsNews, worldnews | six other values, no overlap |

## Milestone 1 plan
- [x] Data loading and audit
- [x] Cleaning pipeline for known shortcuts
- [x] Evaluation protocol (`EVAL.md`)
- [x] Baselines on raw vs. cleaned text (TF-IDF + Logistic Regression / Linear SVM)
- [x] Top-feature inspection and further cleaning
- [x] Second dataset and cross-dataset test
- [x] Findings write-up (`FINDINGS.md`)

## Roadmap
- [x] Baseline + leakage audit + cross-dataset test
- [ ] Transformer comparison
- [ ] Claim extraction + evidence retrieval + verdict
- [ ] Streamlit demo

## Project structure
```
pramaan/
  README.md
  FINDINGS.md      Milestone 1 results and limitations
  EVAL.md          evaluation protocol, results, deviations
  DATA.md          dataset sources and licenses
  requirements.txt
  notebooks/       01_data_audit, 02_baseline, 03_welfake_audit, 04_cross_dataset
  src/             data.py, clean.py, overlap.py, baseline.py
  data/            not committed
```

## Data
The data is not included in this repository. See `DATA.md` for the source,
format and license of each dataset. Place the files here (lowercase folder
names, so it also works on case-sensitive systems):
- `data/raw/isot/Fake.csv` and `data/raw/isot/True.csv`
- `data/raw/welfake/WELFake_Dataset.csv`

## Reproduce
```
python -m venv .venv
.venv\Scripts\activate          # Windows; on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
Run the notebooks in order, 01 to 04.

## Limitations
- ISOT labels describe the source of an article, not whether each claim in
  it is true. Results on ISOT say nothing about fact-checking individual claims.
- WELFake merges other datasets and its label quality was not audited.
- One dataset pair, linear models only. Cleaning removes only the shortcuts
  found so far.
