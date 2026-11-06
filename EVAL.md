# Evaluation Protocol: Pramaan, Milestone 1

Status: Milestone 1 complete. The initial protocol and the ISOT data audit
were written before any trained model was evaluated. Every later change is
logged in section 14.

Items marked TODO are collected in section 15.

## 1. Question
Do simple text classifiers detect misleading news, or do they detect
publisher and formatting artifacts? Specifically:
1. How much of the performance on ISOT disappears when known shortcuts are removed?
2. Does the model still work on a second dataset from different sources?

## 2. Datasets

### Primary: ISOT Fake News Dataset
- Source URL, download date, license: TODO (see section 15)
- Format used: TODO (original CSVs or Excel copies; row counts match the
  published ones)
- Size: 44,898 articles (23,481 fake, 21,417 real).
- Columns: title, text, subject, date. The label column is added by src/data.py.
- Real articles come from Reuters. Fake articles come from websites flagged
  as unreliable by fact-checking organisations (per the dataset's published
  description).
- Data is not redistributed.

### Second: WELFake
- Source and license: TODO (see section 15)
- Size: 72,134 rows. Columns: title, text, label (a serial-number column is
  dropped by src/data.py). It merges other existing datasets.
- Overlap with ISOT: 42,671 of the 68,066 rows long enough to fingerprint
  matched an ISOT article (62.7% of checked rows, 59.2% of all rows). 4,068
  rows were too short to check. Matching uses a mid-article fingerprint
  (characters 100 to 250 of the text with punctuation, spaces and case
  removed), so it catches near-identical copies, not paraphrases.
- External test set: rows that were checked, did not match ISOT, have at
  least 10 words and are de-duplicated: 22,779 articles (13,305 real,
  9,474 fake).
- Data is not redistributed.

## 3. Labels
- Convention: 1 = fake, 0 = real.
- Limitation (ISOT): labels describe the SOURCE (Reuters vs sites flagged as
  unreliable), not an article-by-article fact check. A "fake" label does not
  mean every claim in the article is false. A classifier can therefore score
  well by recognising the publisher. Results on ISOT must not be described
  as "detecting the truth".
- Limitation (WELFake): it merges other datasets, so its labelling method is
  not uniform or fully known to me. Some "real"-labelled articles in the
  error sample contained embedded-tweet text and an opinion tone. Label
  quality was not audited.

## 4. Label mapping
- ISOT is already binary, so no mapping is needed.
- WELFake: label_raw is used as is (0 = real, 1 = fake). Verified on the
  42,671 rows that also appear in ISOT: 21,100 real/real, 21,570 fake/fake,
  1 disagreement. The dataset's own column description says the opposite
  (0 = fake, 1 = real), so the description is wrong or inconsistent. The
  check above was done before the cross-dataset run.

## 5. Data audit findings

### ISOT (notebooks/01_data_audit.ipynb)

| Check | Real | Fake |
|---|---|---|
| Text contains "(Reuters)" | 99.2% | 0.04% |
| Text contains curly quotes | 50.5% | 0.0% |
| Texts under 10 words | 1 | 866 |
| Exact duplicate rows (title + text) | 220 | 5,573 |
| subject values | politicsNews, worldnews | six other values, no overlap |

- No text appears with both labels.
- Rule baseline (no machine learning): predict "fake" when the text has no
  "(Reuters)" tag. Macro-F1 = 0.996 on the full raw dataset.
- Dates: real articles use one format ("December 31, 2017"), fake articles
  use several (for example "19-Feb-18"). After flexible parsing, 0 real and
  10 fake dates are unparsed. Those 10 are malformed scrapes (a URL or a
  headline in the date field). Real date range: 2016-01-13 to 2017-12-31.
  Fake date range: 2015-03-31 to 2018-02-19.
- After cleaning, 2015 contains only fake articles (about 1,574) and real
  articles start in 2016-01. The class balance shifts over time (2016 Q1:
  1,161 real vs 2,644 fake; 2017 Q4: 8,647 real vs 1,093 fake).
- Dates and subject are NOT used as model inputs.

Leftover shortcuts after cleaning B (share of articles in each class):

| Pattern | Real | Fake |
|---|---|---|
| word "Reuters" | 23.0% | 1.2% |
| ticker like (AMZN.O) | 2.5% | 0% |
| URL | 0.2% | 14.4% |
| "Reporting/Editing by" | 1.5% | 0% |
| @handle | 1.3% | 27.3% |

### WELFake (notebooks/03_welfake_audit.ipynb)
- 558 empty titles, 783 empty texts, 9,415 duplicate texts, 1 text that
  appears with both labels.

| Share of articles | label_raw 0 (real) | label_raw 1 (fake) |
|---|---|---|
| "(Reuters)" | 60.7% | 0.0% |
| URL | 0.2% | 13.0% |
| @handle | 9.3% | 20.2% |
| curly quotes | 65.3% | 28.0% |

External set only (non-overlapping rows):

| Share of articles | real | fake |
|---|---|---|
| "(Reuters)" | 0.1% | 0.1% |
| URL | 0.2% | 11.0% |
| @handle | 22.2% | 11.2% |
| curly quotes | 90.2% | 83.3% |
| newspaper-style dateline ("CITY - ") | 21.6% | 0.3% |

The @handle pattern points to fake in ISOT but is twice as common in the
real class of the external set. The dateline pattern is a source marker
inside WELFake, like "(Reuters)" in ISOT.

## 6. Cleaning pipeline (src/clean.py)
B = clean_isot:
1. Drop texts with fewer than 10 words.
2. Drop exact duplicates on (title, text), before any train/test split.
3. Remove the Reuters dateline at the start and any "(Reuters)" tag.
4. Remove quote and apostrophe characters, then normalise whitespace.
5. Drop rows whose date could not be parsed.

Result: 44,898 -> 38,493 rows (real 21,196, fake 17,297).

C = B plus clean_extra, which removes the word "Reuters", exchange tickers,
"Reporting/Editing/Writing by" sign-offs, URLs and @handles.

For the external set, clean_generic repeats steps 1 to 4 without the date
logic (and clean_extra is added for C).

Model input: the text column only.

Known limits:
- Only known shortcuts are removed. Reporting phrases ("said on", "on
  Wednesday") remain.
- Near-duplicates beyond the exact and fingerprint checks were not examined
  inside ISOT.
- Cleaning patterns were chosen by inspecting the whole dataset, based on
  class differences, not tuned to model scores.
- The class balance changed (about 55% real, 45% fake), so splits are
  stratified and macro-F1 is the main metric.

## 7. Splits
- Seed 42 for the main runs. Seeds 0 to 4 for variance runs. Seeds change
  only the ISOT train/test split; the external set stays fixed.
- Random split: 80% train, 20% test, stratified on label, done AFTER cleaning
  and de-duplication.
- Fixed model settings, no tuning. The decision threshold was not tuned on
  the external set.
- Time-based split: dropped (see section 14).
- Cross-dataset: train on the cleaned ISOT training split (vectorizer fitted
  on the training portion only). The same fitted model is scored on the ISOT
  test split and on the WELFake external set.

## 8. Models and reference points
- TF-IDF (1 to 2 word n-grams, min_df 3, max 50,000 features, sublinear tf)
  with Logistic Regression (max_iter 1000) or Linear SVM (defaults),
  random_state 42.

Reference points (no training):
- Majority class on the external set: accuracy 58.4% (13,305 of 22,779),
  macro-F1 about 0.37 (computed by hand from the class counts).
- One-word rule ("no (Reuters)" means fake): macro-F1 0.996 on raw ISOT,
  0.2952 on the external set.

## 9. Experiments and results

| ID | Text used |
|---|---|
| A | Raw text, as downloaded (the naive setup) |
| B | Cleaned text (clean_isot) |
| C | B plus extra markers removed (clean_extra) |

In-domain, ISOT random split, Logistic Regression, macro-F1 (mean +/- std
over 5 seeds):

| ID | Macro-F1 |
|---|---|
| A | 0.9915 +/- 0.0004 |
| B | 0.9849 +/- 0.0013 |
| C | 0.9843 +/- 0.0008 |

B and C are indistinguishable (gap smaller than the spread). A still
contains duplicates, so copies can land in both train and test, and A to B
changes several things at once (duplicates, short texts, Reuters dateline,
quotes). The gap cannot be attributed to shortcuts alone.

Cross-dataset (trained on ISOT, tested on the external set):

| Measure | Result |
|---|---|
| ISOT test, macro-F1, B / C (seed 42) | 0.9834 / 0.9829 |
| External, macro-F1, B / C (seed 42) | 0.5665 / 0.5704 |
| External confusion matrix, B | [[4778, 8527], [1101, 8373]] |
| External confusion matrix, C | [[4862, 8443], [1111, 8363]] |
| External accuracy, B / C | 57.7% / 58.1% |
| External macro-F1, C, Logistic Regression, 5 seeds | 0.5704 +/- 0.0018 |
| External macro-F1, C, Linear SVM, 5 seeds | 0.57 +/- 0.0042 |
| External ROC AUC, C, Logistic Regression / Linear SVM | 0.721 / 0.7532 |

Seed spreads are small because seeds change only the ISOT training split.
They do not measure uncertainty from choosing a different dataset.

Error analysis (C, Logistic Regression): 8,443 real external articles were
labelled fake. Real articles with a newspaper-style dateline were predicted
fake 36.9% of the time (2,876 articles), against 70.8% for those without
(10,429). About 87% of the false positives therefore come from articles
without that dateline.

## 10. Metrics
- Primary: macro-F1.
- Also reported: accuracy, per-class precision and recall, confusion matrix,
  ROC AUC (external set), top 30 features per class.

## 11. Leakage checklist
- [x] Duplicates counted and removed before splitting
- [x] subject column checked (separates the classes, excluded)
- [x] Reuters dateline and tag identified
- [x] Quote-character difference identified
- [x] Near-empty texts identified
- [x] Confirm cleaning worked (quotes gone; "Reuters" word still in 23.0% of
      real articles after B)
- [x] Top features inspected for model B
- [x] Re-train after removing flagged features (experiment C)
- [x] Cross-dataset test
- [x] WELFake overlap with ISOT checked and removed
- [x] AUC and error analysis
- [x] SVM and seed robustness
- [ ] Near-duplicate check inside ISOT (not done)

## 12. Predictions
No predictions were recorded before the experiments were run. This section
was left blank and is not back-filled.

## 13. Success criteria and outcome
1. Experiments A, B and C run with the same protocol and reported with the
   metrics in section 10: met.
2. For C, top features contain no obvious publisher or formatting markers:
   partly met. Web debris was removed, but reporting phrases ("said on",
   "on Wednesday") still lead the real-class features.
3. The cross-dataset result is reported together with its confounds: met.
4. A findings note is written (FINDINGS.md): met.

A high score was NOT a success criterion. Even a high score on C would not
show the model detects truthfulness, because the labels are source-level.

## 14. Deviations
- Section 12 predictions were not recorded before running experiments.
- Time-based split dropped: the class mix shifts strongly over time (2015
  has only fake articles) and the cross-dataset test is the stronger check.
- Seed-variance runs (seeds 0 to 4) added after seeing a 0.0005 gap between
  B and C on a single run.
- The B and C comparison was re-run in separate cells after a truncated
  printout made the earlier outputs ambiguous.
- Date parsing fixed (flexible parser) and a step added to drop rows with
  unparseable dates (B now has step 5).
- WELFake chosen as second dataset. It overlaps ISOT, so only
  non-overlapping rows are used.
- The overlap helper had a bug (missing keys treated as matches) that
  over-counted overlap by 4,068 rows; fixed in src/overlap.py.
- WELFake's documented label direction contradicted the data; the data
  was used (section 4).
- AUC, error analysis and the dateline check were added after seeing the
  cross-dataset result.
- Cleaning frozen after B and C were indistinguishable.

## 15. TODOs
- ISOT: source URL, download date, license (write what the official page
  says; if none is stated, write "not clearly stated"), and whether the
  original CSVs or the Excel copies were loaded.
- WELFake: confirm the source where it was downloaded and its license.
