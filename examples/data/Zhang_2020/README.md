# Zhang et al. (2020) example data

Required local file: `EIS data.zip` (unchanged archive from the cited record)

Source citation:

> Zhang, Yunwei; Tang, Qiaochu; Zhang, Yao; Wang, Jiabin; Stimming, Ulrich; Lee, Alpha A. (2020), “Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning”, Zenodo, Version 1, doi: [10.5281/zenodo.3633835](https://doi.org/10.5281/zenodo.3633835).

Associated article:

> Zhang, Y., Tang, Q., Zhang, Y., Wang, J., Stimming, U. & Lee, A. A. Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning. *Nature Communications* **11**, 1706 (2020), doi: [10.1038/s41467-020-15235-7](https://doi.org/10.1038/s41467-020-15235-7).

The source dataset is published under the [Creative Commons Attribution 4.0 International
licence](https://creativecommons.org/licenses/by/4.0/). The archive is intentionally not tracked in
this repository. Download `EIS data.zip` from the DOI above and place it unchanged in this directory
before running the example notebook. The `Capacity data.zip` file from the same record is not used.

Checksums:

```text
9f31069d770d7d323d8e2bf21267afe2                                  EIS data.zip (MD5, as listed on Zenodo)
b9d0b000bd9a90b1e18a1ce1f5763babbd54f2668c9cd902b505bbe46dfda751  EIS data.zip (SHA256)
8c3d72efa52af2b28942a6c206befce50849d2c655ccc7ffd1f5be204634ed48  EIS data/EIS_state_V_25C01.txt (SHA256, archive member)
```

## Use in the example

[`BFL_Zhang_2020_EIS_example.ipynb`](../../BFL_Zhang_2020_EIS_example.ipynb) reads the archive member
`EIS data/EIS_state_V_25C01.txt`: cell `25C01`, state V, 261 sweeps of 60 points from 20 kHz to
0.02 Hz. The notebook writes two derived input tables to this directory, which are also not tracked:

- `EIS_state_V_25C01_all.txt` — every source row, used for the Lin-KK screen.
- `EIS_state_V_25C01_selected.txt` — the rows of six sweeps (source counters 21, 69, 117, 165, 213
  and 261), used for circuit fitting, comparisons and the AI conditions.

Each derived row is an unchanged source line prefixed with a `Spectrum_ID` column
(`25C01_V_<counter>`), so BFL row references resolve to source lines.

Measurement conditions are declared in the notebook from the article: 45 mAh Eunicell LR2032
LiCoO₂/graphite coin cells; 1C CC–CV charge to 4.2 V and 2C CC discharge to 3.0 V; a 25 °C climate
chamber for cells `25C01`–`25C08`; 5 mA EIS excitation; state V is the 15 min rest after full charge.

Two declarations go beyond what the source states, and the notebook labels both:

- The source `cycle number` counter is declared as `sequence_index`, not `cycle_index`. The authors
  report EIS on even-numbered cycles and capacity checks on odd-numbered cycles, so the counter is
  not assumed to equal the battery cycle number.
- The article does not name the potentiostat. Because every sweep comes from one source file with a
  monotonically increasing `time/s` column, the instrument is declared as one unnamed recording. The
  notebook also shows the comparison gate without this declaration.

The source shows an abrupt decrease of about 0.09 Ω in real impedance between counters 16 and 17,
with no gap in `time/s`. The source records no cause. The detailed analysis therefore starts at
counter 21.
