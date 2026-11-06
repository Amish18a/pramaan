# Pramaan (प्रमाण)
Evidence-first fake news verification. Not just "real or fake": which claims,
what evidence, and how confident.

## Status
Milestone 1 in progress: data audit and cleaning done, baseline classifiers next.
No model results yet.

## Why
A classifier alone can't establish truth. It can learn writing style or
publisher artifacts. This project tests how well simple models generalize
and builds toward evidence-based verification.

## Findings so far (data audit)
In the ISOT dataset the label is almost the same thing as the publisher
(Reuters vs. websites flagged as unreliable), so many shortcuts separate the
classes:

| Check | Real | Fake |
|---|---|---|
| Text contains "(Reuters)" | 99.2% | 0.04% |
| Text contains curly quotes | 50.5% | 0.0% |
| Texts under 10 words | 1 | 866 |
| Exact duplicate rows | 220 | 5,573 |
| `subject` values | politicsNews, worldnews | six other values, no overlap |

- A one-word rule ("no (Reuters) tag means fake") reaches macro-F1 0.996 on
  the raw dataset, with no machine learning. A high score on raw ISOT
  therefore does not show that a model understands truthfulness.
- Real articles use one date format, fake articles use several, so date
  formatting is another possible shortcut. Dates are not used as features.
- Cleaning removes the known shortcuts: 44,898 -> 38,494 rows
  (real 21,196, fake 17,298). See `EVAL.md` for the exact steps.
- Baseline results (TF-IDF + Logistic Regression, random 80/20 split,
  macro-F1, mean ± std over 5 seeds): raw 0.9915 ± 0.0004,
  cleaned 0.9849 ± 0.0013, cleaned + extra patterns 0.9843 ± 0.0008.
- Removing the shortcuts we could identify lowered the score by less than
  one point. The classifier still separates the two sources at about 98.5%,
  mainly using reporting conventions and web boilerplate, not claims.
  Next: test on a second dataset with different publishers.

## Milestone 1 plan
- [x] Data loading and audit
- [x] Cleaning pipeline for known shortcuts
- [x] Evaluation protocol (`EVAL.md`)
- [ ] Baselines on raw vs. cleaned text (TF-IDF + Logistic Regression / Linear SVM)
- [ ] Top-feature inspection and further cleaning
- [ ] Second dataset and cross-dataset test
- [ ] Findings write-up

## Roadmap
- [ ] Baseline + leakage audit + cross-dataset test
- [ ] Transformer comparison
- [ ] Claim extraction + evidence retrieval + verdict
- [ ] Streamlit demo

## Project structure
```
pramaan/
  README.md
  EVAL.md          evaluation protocol
  DATA.md          dataset sources and licenses
  requirements.txt
  notebooks/       01_data_audit.ipynb
  src/             data.py (loading), clean.py (cleaning)
  data/            not committed
```

## Data
The data is not included in this repository. See `DATA.md` for the source,
format and license of each dataset. Place `Fake.csv` and `True.csv` in
`data/raw/isot/`.

## Reproduce
```
python -m venv .venv
.venv\Scripts\activate          # Windows; on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
Then open `notebooks/01_data_audit.ipynb`.

## Limitations
- ISOT labels describe the source of an article, not whether each claim in
  it is true. Results on ISOT say nothing about fact-checking individual claims.
- Cleaning removes only the shortcuts found so far. Others may remain.

