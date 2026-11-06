# Pramaan: Milestone 1 findings

Status: baseline experiments complete (notebooks 01 to 04). Every number
below comes from those notebooks. Full protocol and deviations: EVAL.md.

## Question
Do simple text classifiers detect misleading news, or do they recognise the
publisher and formatting of the sources they were trained on?

## Setup
- ISOT: 44,898 articles (23,481 fake, 21,417 real). After cleaning: 38,493.
- WELFake (second dataset): 72,134 rows. About 59% of rows matched ISOT
  articles (mid-article fingerprint) and were removed. External test set:
  22,779 articles (13,305 real, 9,474 fake): checked, non-overlapping, at
  least 10 words, de-duplicated.
- Model: TF-IDF (1 to 2 grams, min_df 3, 50,000 features, sublinear tf)
  with Logistic Regression or Linear SVM. 80/20 stratified split, seed 42.
- Cleaning: B = duplicates, near-empty texts, Reuters dateline and quote
  characters removed. C = B plus the word "Reuters", tickers, sign-offs,
  URLs and @handles removed.

## Results

In-domain, ISOT random split (macro-F1, mean +/- std over 5 seeds):

| Text | Macro-F1 |
|---|---|
| A: raw | 0.9915 +/- 0.0004 |
| B: cleaned | 0.9849 +/- 0.0013 |
| C: cleaned + extra | 0.9843 +/- 0.0008 |

Cross-dataset: trained on ISOT, tested on the WELFake external set:

| Measure | Result |
|---|---|
| Macro-F1, B (seed 42) | 0.5665 |
| Macro-F1, C (seed 42) | 0.5704 |
| Macro-F1, C, Logistic Regression (5 seeds) | 0.5704 +/- 0.0018 |
| Macro-F1, C, Linear SVM (5 seeds) | 0.57 +/- 0.0042 |
| ROC AUC, C, Logistic Regression / Linear SVM | 0.721 / 0.7532 |
| Accuracy, C, Logistic Regression | 58.1% (always predicting "real": 58.4%) |
| Real articles labelled fake, C | 8,443 of 13,305 |
| Fake articles caught, C | 88% |
| One-word rule ("no (Reuters)" means fake): ISOT raw / external | 0.996 / 0.2952 |

## What we found
1. ISOT labels describe the source (Reuters vs sites flagged as unreliable).
   Many shortcuts separate the classes: the Reuters tag (99.2% of real vs
   0.04% of fake), curly quotes (50.5% vs 0.0%), URLs and @handles in fake
   articles, and the subject column (no overlap between classes).
2. Removing the shortcuts we could name cost less than one point in-domain.
   The model still separates the two ISOT sources at about 98%, mainly
   through reporting conventions and web boilerplate.
3. On articles from other sources, the same models fall to about 0.57
   macro-F1, about the level of always answering "real". The drop holds for
   both Logistic Regression and Linear SVM.
4. Some signal transfers: AUC of 0.72 to 0.75 means a random fake article
   gets a higher fake score than a random real one about three times in
   four. But the decision threshold learned on ISOT does not transfer: most
   real external articles are labelled fake.

## Error analysis
I read 8 random real WELFake articles that the model labelled fake: four
read like conventional news or feature reporting, two are opinion or
first-person pieces, and two contain embedded-tweet text ("pic. twitter").
In ISOT that embedded-tweet pattern pointed to the fake class, yet WELFake
labels these real. Eight samples cannot show that WELFake's labels are noisy
or source-level, but they are a reason for caution.

Dateline check: real WELFake articles with a newspaper-style dateline (place
name followed by an em dash) were predicted fake 36.9% of the time (2,876
articles), versus 70.8% for real articles without one (10,429). So this
pattern accounts for a small part of the errors: about 87% of the false
positives come from articles without it. The same pattern appears in only
0.3% of the fake class, so WELFake has its own source marker, as ISOT does
with "(Reuters)".

Not confirmed: the broader hypothesis that the model learned the style of
ISOT's sources, so that legitimate articles in a different house style look
suspicious. The dateline check does not confirm it.

## What we cannot conclude
- The cause of the drop. Publisher style, topic, time period and label
  quality all differ between ISOT and WELFake, and this experiment cannot
  separate them.
- That WELFake's labels are reliable. It merges other datasets, and label
  quality was not audited.
- That results hold beyond these two datasets and two linear models.
- Seed spreads are small because seeds change only the ISOT training split;
  the external set is fixed. They do not measure uncertainty from choosing
  a different dataset.
- That the A to B drop (0.9915 to 0.9849) is due to shortcuts alone: it also
  reflects removing duplicates and near-empty texts.

## Implication for Pramaan
A classifier trained on one set of sources learns those sources. Judging
whether a claim is supported needs evidence from outside the article. This
motivates the next milestones: any transformer comparison must be evaluated
cross-dataset, and the verdict should come from claims checked against
retrieved evidence, with the text classifier as one signal among several.

## Limitations of this milestone
- Single dataset pair, linear models only, one cleaning design.
- Cleaning patterns were chosen by inspecting the whole dataset, based on
  class differences, not tuned to model scores.
- The time-based split in the original plan was dropped (the class mix
  shifts strongly over time; the cross-dataset test is the stronger check).
- No predictions were recorded before the experiments were run.
