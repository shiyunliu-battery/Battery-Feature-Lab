# Examples

Each example runs one public dataset through BFL, inspects the compact result, retrieves evidence,
and optionally asks GPT-5.6 to interpret the saved JSON under four controlled input conditions.

| Notebook | Modality | Dataset | BFL entry point |
|---|---|---|---|
| [`BFL_Catenaro_Onori_example.ipynb`](BFL_Catenaro_Onori_example.ipynb) | Time series | Catenaro & Onori (2021), galvanostatic discharge of an NCA cell | `bfl.analyze()` |
| [`BFL_Zhang_2020_EIS_example.ipynb`](BFL_Zhang_2020_EIS_example.ipynb) | EIS | Zhang et al. (2020), 261 state-V spectra of an ageing LiCoO₂/graphite coin cell | `bfl.analyze_eis()` |

## Layout

```text
examples/
├── BFL_<dataset>_example.ipynb   # one notebook per dataset
├── data/<dataset>/README.md      # citation, licence, download and checksums
└── outputs/<dataset>/<run>/      # derived JSON artifacts from the notebook
```

Source data is not tracked. Follow each dataset README to download it into `data/<dataset>/`.
Derived JSON artifacts are tracked so the results can be inspected without running the notebook.
Normalized Parquet files and the derived EIS input tables are excluded.

## Running

```console
uv sync --python 3.12 --extra ai --extra eis
uv run --with jupyter jupyter lab examples/
```

The AI sections are optional. They read `OPENAI_API_KEY` from the environment or ask for it with
a hidden prompt; the key is never written into a notebook. Uploaded files are deleted from the
OpenAI Files API after each response, and requests use `store=False`.

## Input conditions for the AI sections

| Condition | Content |
|---|---|
| 1. Raw measurements | The measurements BFL analysed, unchanged |
| 2. Preprocessing report only | The BDS conversion report or EIS intake report |
| 3. Preprocessing + BFL JSON | The report and every BFL JSON artifact |
| 4. BFL-guided retrieval | Compact results plus the evidence records they reference |

Model output varies between runs. Treat the saved responses as one sample per condition, not as a
benchmark.

Saved notebook outputs record the software versions that produced them. The Zhang et al. example
was run with BFL 0.6.0 and eisfit 0.1.1; re-running it with a later release regenerates the
outputs, for example adding the multistart near-best ranges reported from eisfit 0.2.0.
