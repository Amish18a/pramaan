# Data sources

Datasets are not redistributed in this repository. Download them from the
sources below and place them as described in README.md.

| Dataset | Source link | Size | Labels | License | Notes |
|---------|-------------|------|--------|---------|-------|
| ISOT Fake News | TODO: official page | 44,898 articles (23,481 fake, 21,417 real) | fake / real (loader uses 1 = fake, 0 = real) | TODO: as stated on the official page, or "not clearly stated" | Real articles from Reuters; fake from sites flagged as unreliable (per published description). Label is source-level. Not redistributed. |
| WELFake | TODO: confirm download source (Zenodo record 4561253 is the dataset's published record: https://zenodo.org/record/4561253) | 72,134 rows | label_raw: 0 = real, 1 = fake (verified on rows shared with ISOT; the dataset's own description says the opposite) | TODO: as stated on the source page | Merges other datasets. About 59% of rows match ISOT articles; only 22,779 non-overlapping rows are used as the external test set. Label quality not audited. Not redistributed. |

## Citing
If you publish results, cite the original dataset papers and sources.
TODO: add the citation for each dataset from its official page.
