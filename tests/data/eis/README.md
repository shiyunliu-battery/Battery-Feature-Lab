# EIS regression fixture

`ID01.csv` is copied unchanged from the `EIS_dataset/ID01.csv` example in
[Battery-EIS-Fit](https://github.com/shiyunliu-battery/Battery-EIS-Fit/tree/v0.1.1/EIS_dataset).

Original dataset: M. Moertelmaier, M. Kasper and S. Clark, **EIS data of 54
21700 cells**, Zenodo, 26 May 2025. DOI:
[10.5281/zenodo.15422339](https://doi.org/10.5281/zenodo.15422339).

The original dataset is licensed under **CC BY 4.0**. This fixture retains that
license and attribution; the BFL software license does not replace it.

The test declares only the source identifier `ID01`. Conditions absent from
the CSV are not inferred. The test checks the released `eisfit 0.1.1` interface,
Lin-KK execution, bounded fit error and evidence output. It does not establish
parameter identifiability or a degradation mechanism.
