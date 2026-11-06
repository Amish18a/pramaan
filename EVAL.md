# Evaluation Protocol: Pramaan, Milestone 1

Status: draft v1, written before any trained model was evaluated.
Rule: any change made to this protocol after seeing test results must be
logged in the "Deviations" section at the bottom, with the reason.

Items marked TODO must be filled in by me.

## 1. Question
Do simple text classifiers detect misleading news, or do they detect
publisher and formatting artifacts? Specifically:
1. How much of the performance on ISOT disappears when known shortcuts are removed?
2. Does the model still work on a second dataset from different sources?

## 2. Datasets

### Primary: ISOT Fake News Dataset
- Source URL: TODO
- Download date: TODO
- Format used: CSV (Fake.csv, True.csv) in data/raw/isot/
- Size: 44,898 articles (23,481 fake, 21,417 real). Matches the published counts.
- Columns: title, text, subject, date. The label column is added by src/data.py.
- Real articles come from Reuters. Fake articles come from websites flagged
  as unreliable by fact-checking organisations (per the dataset's published
  description. TODO: confirm against the official page).
- License: TODO (write exactly what the official page says; if it is not
  clearly stated, write "not clearly stated"). Data is not redistributed.

### Second dataset
- To be chosen by week 3, before looking at any cross-dataset results.
- Criteria: article-level (not single sentences), different publishers
  from ISOT, license checked and recorded in DATA.md.

## 3. Labels
- Convention: 1 = fake, 0 = real.
- Limitation: ISOT labels describe the SOURCE (Reuters vs sites flagged as
  unreliable), not an article-by-article fact check. A "fake" label does not
  mean every claim in the article is false. A classifier can therefore score
  well by recognising the publisher. Results on ISOT must not be described
  as "detecting the truth".

## 4. Label mapping
- ISOT is already binary, so no mapping is needed.
- For any second dataset with a different label scheme, the mapping and the
  information it throws away will be written here BEFORE evaluation.

## 5. Data audit findings (ISOT)
From notebooks/01_data_audit.ipynb.

| Check | Real | Fake |
|---|---|---|
| Text contains "(Reuters)" | 99.2% | 0.04% |
| Text contains curly quotes | 50.5% | 0.0% |
| Texts under 10 words | 1 | 866 |
| Exact duplicate rows (title + text) | 220 | 5,573 |
| subject values | politicsNews, worldnews | six other values, no overlap |

- No text appears with both labels.
- Rule baseline (no machine learning): predict "fake" when the text has no
  "(Reuters)" tag. Macro-F1 = 0.996 on the full raw dataset. This is the
  bar any model on raw text has to be compared against.
- Dates: real articles use one format ("December 31, 2017"), fake articles
  use several (for example "19-Feb-18"). After flexible parsing, 0 real and
  10 fake dates are still unparsed. Real date range: 2016-01-13 to
  2017-12-31. Fake date range: 2015-03-31 to 2018-02-19.
- Dates and subject will NOT be used as model inputs.

## 6. Cleaning pipeline (src/clean.py, function clean_isot)
1. Drop texts with fewer than 10 words.
2. Drop exact duplicates on (title, text), before any train/test split.
3. Remove the Reuters dateline at the start and any "(Reuters)" tag.
4. Remove quote and apostrophe characters, then normalise whitespace.

Result: 44,898 -> 38,494 rows (real 21,196, fake 17,298).
Model input: the text column only.

Known limits:
- Only known shortcuts are removed. Others (for example sign-offs or
  boilerplate) may remain and will be looked for in the top-feature check.
- Near-duplicates (almost identical, not exact) are not removed yet.
- The class balance changed (about 55% real, 45% fake), so splits are
  stratified and macro-F1 is the main metric.

## 7. Splits
- Seed: 42 for every random operation.
- Random split: 80% train, 20% test, stratified on label, done AFTER cleaning
  (and after de-duplication).
- First pass uses fixed model settings, no tuning, so the test set cannot
  influence settings. If tuning is added later, it uses 5-fold stratified
  cross-validation on the training portion only.
- The test set is evaluated once per experiment configuration.
- Time-based split: cutoff date TODO. Choose it after checking monthly class
  counts so that both classes appear in train and test. Caveat: real articles
  cover a narrower date range than fake ones, so a time split partly mixes
  class with time. The 10 rows with no date are left out of this split only.
- Cross-dataset: train on the ISOT training split, test on the second dataset.

## 8. Models and reference points
Reference points (no training):
- Majority-class predictor
- One-word rule ("no (Reuters)" means fake): macro-F1 0.996 on raw data

Models:
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
Same settings across all experiments below.

## 9. Experiments (ablation)
| ID | Text used |
|---|---|
| A | Raw text, as downloaded (the naive setup) |
| B | Cleaned text (src/clean.py) |
| C | Cleaned text, plus extra markers removed after the top-feature check on B |

Everything else stays the same (model, seed, split procedure). The gap
between A and B estimates how much of the score came from shortcuts.
Note: A still contains duplicates, so copies can land in both train and
test. This is part of what A is meant to show.

## 10. Metrics
- Primary: macro-F1 (F1 per class, averaged, so the larger class cannot
  hide a weak one).
- Also report: accuracy, per-class precision and recall, confusion matrix.
- For every model: the 30 highest-weighted features per class.

## 11. Leakage checklist
- [x] Duplicates counted and removed before splitting
- [x] subject column checked (separates the classes, excluded)
- [x] Reuters dateline and tag identified
- [x] Quote-character difference identified
- [x] Near-empty texts identified
- [ ] Confirm cleaning worked (leftover "Reuters" mentions, quotes, endings)
- [ ] Top features inspected for model B
- [ ] Re-train after removing flagged features (experiment C)
- [ ] Near-duplicate check
- [ ] Cross-dataset test

## 12. Predictions (write BEFORE running any model)
- Expected macro-F1 for A: TODO
- Expected macro-F1 for B: TODO
- Expected macro-F1 for C: TODO
- Share of A's score I think comes from shortcuts, and why: TODO

## 13. Success criteria
The milestone succeeds if:
1. Experiments A, B and C are run with the same protocol and reported with
   the metrics in section 10.
2. For C, the top features contain no obvious publisher or formatting
   markers, or the remaining ones are documented.
3. The cross-dataset result is reported together with its confounds.
4. A findings note is written in the README.

A high score is NOT a success criterion. Even a high score on C does not
show the model detects truthfulness, because the labels are source-level.

## 14. Deviations
(none yet)


## 15. TODOs
- Source URL, download date, license: copy from the official page. If no license is stated, write “not clearly stated”.