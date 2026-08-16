# Five-paper method mapping for short-term BFL records

Date: 2026-08-16

## Scope and decision rule

This note reviews the five PDFs supplied in `docs/research/` and asks a narrow question: which
deterministic, source-grounded time-series descriptors can be transferred into these four BFL record
families?

- `operation.window_summary`
- `response.capacity_aligned_profile`
- `response.current_step`
- `response.relaxation_signature`

The verdict vocabulary is strict:

- **ADOPT**: the paper specifies an observable calculation that BFL can recompute from its source
  interval without a trained model or an unsupported physical inference.
- **REFERENCE ONLY**: the paper supports the representation or QA design, but not a transferable
  battery calculation.
- **REJECT**: the method requires training labels, pack-only semantics, prose generation, or an
  inference that a short single-cell record cannot support.

The overall result is intentionally conservative. Of the five papers, only TimeSeries2Report (TS2R)
contains directly reusable short-window descriptive calculations. None of the five establishes the
electrochemical method for a capacity-aligned charge/discharge profile, a current-step resistance, or
a post-load relaxation response. Those response records therefore retain their existing primary
sources and provider gates; this review does not manufacture new validation for them.

## Paper identity, data, and reproducibility

| PDF | Identity and status | Data actually used | Method or available tool | Overall decision |
|---|---|---|---|---|
| [arXiv:2512.16453v2](https://arxiv.org/abs/2512.16453) | Yang et al., *TimeSeries2Report prompting enables adaptive large language model management of lithium-ion batteries*. The preprint declares no publisher DOI. | Laboratory subsets: 6 selected LFP cells from a 124-cell MIT source and 4 selected NCA cells from a 130-cell TJU source, 10 complete cycles per selected cell, four 100-point operating sections per cycle, sampled at 5 or 10 s ([PDF pp. 19-20, Supplementary Note 1 and Table S1](https://arxiv.org/pdf/2512.16453#page=19)). Field data: 28 modules with 16 series cells per module, one-minute sampling over six months and 259,778 time steps per module ([PDF p. 24, Supplementary Note 4 and Table S2](https://arxiv.org/pdf/2512.16453#page=24)). The main-text statement about 1,792 samples "each comprising 259,778 minutes" is ambiguous and should not be used as a sizing fact ([PDF p. 2](https://arxiv.org/pdf/2512.16453#page=2)). | Deterministic slicing and rule-based descriptors precede the LLM. The paper says code and data are open-sourced but gives no repository or archive identifier in the PDF, so the artifact cannot be selected as a BFL dependency from this document alone. | **ADOPT in part**: reuse selected numerical window primitives, not the generated prose, LLM, anomaly/SOC prediction, or charge-management decisions. |
| [arXiv:2512.24686v1](https://arxiv.org/abs/2512.24686) | Zhou et al., *BatteryAgent: Synergizing Physics-Informed Interpretation with LLM Reasoning for Intelligent Battery Fault Diagnosis*. The preprint declares no publisher DOI. | More than 690,000 charging segments from 347 vehicles: 292 labelled normal and 55 fault-labelled. BMS voltage, current, and temperature are sampled every 10 s; the split is by vehicle ([PDF p. 4, Section III.A-B](https://arxiv.org/pdf/2512.24686#page=4)). | Ten engineered features feed LightGBM, SHAP, and an LLM fault-reasoning layer. No code or model archive is identified in the PDF. | **REFERENCE ONLY for a few observables; REJECT the diagnostic pipeline.** The trained classifier, SHAP attribution, expert fault matrix, token-probability calibration, root-cause text, and maintenance advice are outside BFL. |
| [arXiv:2606.12481v1](https://arxiv.org/abs/2606.12481) | Kim et al., *Representing Time Series as Structured Programs for LLM Reasoning*. The preprint declares no publisher DOI. | A synthetic TSEdit benchmark; 15 curated ETTh1 oil-temperature examples; Wafer and ECG200 classification data; and generic TSQA, TRQA, and ETI question-answering benchmarks ([PDF pp. 6-8 and p. 13, Table 5](https://arxiv.org/pdf/2606.12481#page=6)). There is no battery dataset. | T2SP decomposes a series into a B-spline trend, Fourier-selected sinusoids, spike/Gaussian events, and a residual, then serializes the components as a program ([PDF pp. 4-5, Section 3](https://arxiv.org/pdf/2606.12481#page=4)). The PDF contains no code URL and says the synthetic dataset will be released after acceptance ([PDF p. 12, Appendix B.1](https://arxiv.org/pdf/2606.12481#page=12)). | **REFERENCE ONLY for structured representation; REJECT the decomposition as a BFL extractor.** |
| [doi:10.1039/D6DD00028B](https://doi.org/10.1039/D6DD00028B) | Lee et al., *Structured domain knowledge enables trustworthy materials science question-answering with large language models*, *Digital Discovery* 5, 2243-2253 (2026). | 11,027 water-splitting papers are filtered and processed into 2,343 structured records. Evaluation uses 202 DOI-identification questions, 202 descriptive questions, and 50 numerical-property questions (PDF pp. 3 and 5-7). It is literature data, not battery time-series data. | A three-stage literature pipeline creates hierarchical JSON, followed by query reformulation and hybrid dense/sparse retrieval. Code and data are archived at [doi:10.5281/zenodo.19676935](https://doi.org/10.5281/zenodo.19676935) (PDF p. 10, Data availability). | **REFERENCE ONLY for schema, condition binding, provenance, and evaluation design.** It is not a time-series feature provider. |
| [doi:10.1016/j.xinn.2025.101091](https://doi.org/10.1016/j.xinn.2025.101091) | Chen et al., *Advancing battery research through large language models: A review*, *The Innovation* 7(2), 101091 (2026). | A narrative review spanning battery knowledge integration, materials, manufacturing, and system management. It has no new time-series benchmark or reproducible extractor (PDF p. 2, abstract and scope). | Survey of LLM applications, RAG, time-series models, and model/tool integration. No BFL-callable method is introduced. | **REFERENCE ONLY for the product boundary; REJECT all reviewed predictive claims as feature definitions.** |

### Four-dimension mapping at a glance

| Paper | `operation.window_summary` | `response.capacity_aligned_profile` | `response.current_step` | `response.relaxation_signature` |
|---|---|---|---|---|
| TS2R | **ADOPT in part:** numerical window shape and adjacent-descriptor merge, with new elapsed-time and QA rules | **REFERENCE ONLY:** capacity is merely another time channel | **REFERENCE ONLY:** local-slope change may locate a candidate, but is not a response calculation | **ADOPT shape only:** only after BFL has independently gated a rest |
| BatteryAgent | **REFERENCE ONLY / limited observable reuse:** CC/CV duration ratio, endpoints, voltage slope, and thermal rate | **REJECT:** no paired phase or capacity coordinate | **REJECT:** no step or pulse method | **REJECT:** no rest method |
| T2SP | **REFERENCE ONLY:** structured representation; **REJECT** its generic decomposition | **REJECT** | **REJECT:** generic spike/Gaussian events are not current steps | **REJECT:** no rest semantics or elapsed-time response |
| Structured domain knowledge Q/A | **REFERENCE ONLY:** JSON hierarchy, condition binding, units, and source identity | **REFERENCE ONLY at the record-envelope level** | **REFERENCE ONLY at the record-envelope level** | **REFERENCE ONLY at the record-envelope level** |
| LLM battery review | **REFERENCE ONLY:** supports tool-mediated, structured input to downstream LLMs | **REFERENCE ONLY:** no primary method | **REFERENCE ONLY:** no primary method | **REFERENCE ONLY:** no primary method |

## Exact transferable content by BFL dimension

### 1. `operation.window_summary`

#### ADOPT from TS2R, with BFL-specific QA

TS2R separates its deterministic signal analysis from later prose generation. It first computes
initial/final value, mean, standard deviation, a linear trend, local-slope transitions, detrended
variance, and outliers; adjacent windows with the same descriptor are merged ([PDF pp. 12-14,
Sections 4.1-4.6 and Tables 1-2](https://arxiv.org/pdf/2512.16453#page=12)). These are useful machine-readable
window primitives:

- `initial_value` and `final_value`, with exact source rows and elapsed times;
- a numerical time slope plus a categorical direction derived from a declared threshold;
- transition candidates from changes between adjacent local slopes;
- detrended fluctuation magnitude, retaining the numeric variance rather than only "slight" or
  "noticeable";
- isolated-outlier count, magnitude, and interval;
- merge of adjacent descriptor segments only when their method, threshold, and quality state match.

Do not copy TS2R's fixed ten-sample slice. Ten points mean 50 s in one laboratory source, 100 s in
another, and 10 minutes in its field source. BFL must use elapsed-time windows and preserve irregular
sampling. Time-weighted means, variation, and exposure remain previous-ZOH calculations; OLS trend or
local-slope calculations must use actual seconds, never sample index.

Do not copy TS2R's thresholds either. Its window is fixed at ten timestamps, and its trend and
transition sensitivities are empirically chosen per variable and dataset from a global range
`Delta = x_max - x_min`; the stated values vary across MIT, TJU, and ZJU data ([PDF p. 28,
Supplementary Note 6 and Table S3](https://arxiv.org/pdf/2512.16453#page=28)). Global range is unstable in the
presence of an outlier and leaks information across windows. BFL should instead record thresholds
based on measurement resolution/noise and local temporal support, with minimum point count, finite
coverage, gap, and quantisation gates.

#### Limited reference from BatteryAgent

BatteryAgent's CC fraction `t_CC / (t_CC + t_CV)`, initial voltage, voltage-time slope, maximum
temperature rate, and terminal temperature are directly observable summaries ([PDF p. 3, Section
II.B and Table I](https://arxiv.org/pdf/2512.24686#page=3)). They can inform field names in
`operation.window_summary`, but their paper-assigned meanings such as degradation, overcharge, fault,
or thermal-runaway risk must not be transferred. Those meanings are conditioned on its vehicle
labels, expert fault matrix, and trained model.

Its pack-to-cell voltage ratio, inter-cell voltage correlation, and maximum cell-temperature spread
are inapplicable to a single-cell input. They are admissible only in a future pack record after BDS
provides explicit channel identity, topology, and simultaneity semantics.

The diagnostic benchmark also uses coarse vehicle labels: every charging segment from a fault-labelled
vehicle is labelled abnormal regardless of degradation stage ([PDF p. 5, Section IV.B](https://arxiv.org/pdf/2512.24686#page=5)).
That label design cannot validate a per-window fault or health interpretation in BFL.

#### Representation evidence from the other three papers

T2SP supports the high-level choice to expose compact structure instead of sending raw arrays to an
LLM, but its B-spline/Fourier/Gaussian decomposition should not be run on cycler data. The authors
explicitly state that a generic trend-period-event abstraction is not optimal for domain-specific
morphology ([PDF p. 9, Limitations](https://arxiv.org/pdf/2606.12481#page=9)). On a battery record it could smooth
away protocol steps, interpret a protocol as seasonality, or bridge a gap.

The Digital Discovery paper provides stronger evidence for keeping conditions and relations inside
the structured record. Its hierarchical JSON improves retrieval by allowing section/method filtering
and by binding facts to their experimental context ([PDF p. 8, Discussion](https://doi.org/10.1039/D6DD00028B)).
It also identifies non-normalized units and lost experimental relationships as causes of retrieval
mismatch ([PDF p. 9, Limitations](https://doi.org/10.1039/D6DD00028B)). BFL already has the right response:
canonical units upstream, `source_intervals`, `method`, `reference_frame`, applicability, quality,
and interpretation limits in every record.

The Innovation review is not extraction evidence. Its useful boundary is that text-native LLMs do
not handle time-series reliably and that an LLM should invoke external physical or analytical tools
rather than simulate the calculation itself ([PDF pp. 8 and 11, Battery system management](https://doi.org/10.1016/j.xinn.2025.101091)).

### 2. `response.capacity_aligned_profile`

**No method in these five papers is adoptable for this record.**

- TS2R sometimes treats charge and discharge capacity as ordinary time channels, but it never pairs
  charge/discharge phases on a common capacity coordinate, states an interpolation rule, or computes
  `V_charge(q) - V_discharge(q)`.
- BatteryAgent has neither a discharge pair nor a capacity-coordinate method.
- T2SP is a generic time-axis decomposition; mapping its components onto capacity would add an
  unsupported abstraction.
- The Digital Discovery paper and The Innovation review contain no applicable calculation.

Decision: **REFERENCE ONLY** for the general idea of structured numerical evidence; **REJECT** every
attempt to cite these papers as validation of capacity alignment, apparent paired polarisation, or a
health/lifetime conclusion. Keep the completeness, monotonicity, common-support, comparable-current,
temperature, denominator, and interpolation gates defined in
[`short_term_features.md`](short_term_features.md#2-capacity-aligned-terminal-profile-responsecapacity_aligned_profile).

### 3. `response.current_step`

**No method in these five papers is a current-step response extractor.**

TS2R's local-slope transition can be used only as a candidate locator. It does not require a stable
pre-step rest, a significant `delta_current`, a stable plateau, a response time contained inside the
same event, or a `delta_voltage / delta_current` calculation. T2SP's spike/Gaussian events are even
less appropriate because they impose generic shapes after trend/period removal. BatteryAgent uses
whole-charge slopes and pack statistics, not pulse transients.

Decision: **REFERENCE ONLY** for candidate segmentation, followed by BFL's electrical eligibility
gates. **REJECT** generic transition or event amplitude as a substitute for `delta_current`, measured
voltage response, response time, apparent DC resistance, or PyProBE pulse resistance. A provider
failure must remain `provider_error`; none of these papers licenses a fallback of the same name.

### 4. `response.relaxation_signature`

TS2R contributes only reusable shape primitives after BFL has independently proved that the source
interval is a rest: initial/final voltage, numerical trend, detrended variation, outliers, and merging
of adjacent intervals with equivalent descriptors. Its curated sections include load-to-idle
transitions, but it does not define rest current, fixed elapsed-time voltage checkpoints, equilibrium,
or a relaxation-specific physical model ([PDF pp. 19-20, Supplementary Note 1](https://arxiv.org/pdf/2512.16453#page=19)).

Decision: **ADOPT shape primitives only**, calculated on actual elapsed seconds after the rest gate.
Keep the fixed-time, no-extrapolation, temporal-support, temperature, and source-predecessor rules from
[`short_term_features.md`](short_term_features.md#4-relaxation-signature-responserelaxation_signature).
Do not infer OCV, SOC, capacity, SOH, lithium plating, or a mechanism from these descriptors.

BatteryAgent, T2SP, the Digital Discovery paper, and The Innovation review provide **no transferable
relaxation method**. Their trend, event, Q/A, or prediction layers are therefore **REFERENCE ONLY** at
the representation level and **REJECTED** as relaxation calculations.

## Minimal record changes supported by this review

This paper set supports a small addition to the existing four-family design, not a new framework:

1. Add optional numerical shape fields to `operation.window_summary`: actual-time slope,
   resolution-aware direction, detrended variation, transition candidates, isolated outliers, and
   merged descriptor intervals. Preserve the existing previous-ZOH exposure statistics.
2. Reuse those shape primitives inside an already eligible `response.relaxation_signature`; do not
   use them to decide that an interval is a rest.
3. Make experimental context relational and queryable. Every derived value should retain the source
   interval, provider/method/parameters, unit, reference frame, channel mapping, finite/gap coverage,
   failed gates, and interpretation limits.
4. Keep `response.capacity_aligned_profile` and `response.current_step` on their existing
   electrochemical/provider methods. These five PDFs add no replacement algorithm.
5. Emit JSON only. Natural-language rendering, RAG, vector indexing, Q/A, and report generation belong
   downstream.

## Explicit rejections

The following must not enter BFL on the strength of these papers:

- TS2R's LLM-written report, FactScore judge, SOC forecast, anomaly decision, or remaining-charge-time
  forecast;
- a universal ten-sample window or thresholds scaled from the complete dataset range;
- BatteryAgent's LightGBM, SHAP, feature-fault matrix, token likelihood, fault severity, root-cause
  explanation, or maintenance recommendation;
- pack inconsistency or thermal-gradient metrics when channel topology is absent;
- T2SP B-spline/Fourier/spike/Gaussian decomposition of cycler data;
- the Digital Discovery paper's NLP extraction or RAG stack as a time-series provider;
- any SOH, ageing, lifetime, safety, fault, chemistry, or protocol-name inference from a short record;
- any direct transfer of claims summarized by The Innovation review without returning to and
  validating the cited primary method.

## Confidence in the evidence

- **Highest for representation QA:** the peer-reviewed Digital Discovery paper provides code/data and
  an evaluated comparison between raw and structured information, but its domain is literature Q/A.
- **Useful but provisional for window primitives:** TS2R specifies deterministic calculations and
  battery datasets, but it is a preprint and its thresholds are explicitly empirical and
  dataset-specific.
- **Not sufficient for product feature adoption:** BatteryAgent and T2SP are preprints with no
  identified code archive in the supplied PDFs; their central claims depend respectively on a trained
  fault pipeline or non-battery benchmarks.
- **Context only:** The Innovation article is a peer-reviewed review, not a primary validation of any
  one extractor.

This evidence level is why the adopted scope is limited to recomputable descriptive primitives and
why all learned or mechanistic interpretations remain outside BFL.
